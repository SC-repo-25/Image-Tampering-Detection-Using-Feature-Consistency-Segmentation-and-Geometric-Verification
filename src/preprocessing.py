"""
src/preprocessing.py
=====================
MODULE 1: Image Input & Preprocessing

RESPONSIBILITIES (per project spec)
------------------------------------
- validate image
- resize where necessary
- normalize
- convert color spaces
- noise reduction
- preserve original image
- generate preprocessing statistics

WHERE THIS SITS IN THE PIPELINE
--------------------------------
Input Image -> [THIS MODULE: Validation -> Preprocessing] -> Histogram Analysis -> ...

CONCEPTS DEMONSTRATED (syllabus mapping)
------------------------------------------
- Module 1 (Image formation & filtering): color space conversion (RGB<->Gray,
  RGB<->YCrCb), Gaussian filtering for noise reduction, normalization.

DESIGN NOTE ON "PRESERVE ORIGINAL IMAGE"
------------------------------------------
Tampering forensics is destructive if you're not careful: e.g. resizing an
image changes its noise/JPEG-artifact statistics, which later modules may
rely on. So this module NEVER mutates the original array in place. Every
function returns a new array, and PreprocessingResult keeps a reference to
the untouched original alongside the processed version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import PreprocessingConfig

logger = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    """Raised when an input image fails validation checks.

    Using a dedicated exception (rather than a generic ValueError) lets
    main.py catch preprocessing-specific failures and print a clean CLI
    error message instead of a raw traceback, per development rule about
    error handling.
    """


@dataclass
class PreprocessingStats:
    """Numerical record of what preprocessing did to the image.

    This is printed to the CLI ("Preprocessing completed:") and saved into
    the machine-readable JSON report, so every number here must be REAL,
    computed from the actual image -- never a placeholder.
    """

    original_width: int
    original_height: int
    original_channels: int
    processed_width: int
    processed_height: int
    was_resized: bool
    resize_scale_factor: float
    mean_intensity_before: float
    mean_intensity_after: float
    std_intensity_before: float
    std_intensity_after: float

    def to_dict(self) -> dict:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "original_channels": self.original_channels,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "was_resized": self.was_resized,
            "resize_scale_factor": round(self.resize_scale_factor, 4),
            "mean_intensity_before": round(self.mean_intensity_before, 4),
            "mean_intensity_after": round(self.mean_intensity_after, 4),
            "std_intensity_before": round(self.std_intensity_before, 4),
            "std_intensity_after": round(self.std_intensity_after, 4),
        }


@dataclass
class PreprocessingResult:
    """Everything downstream modules need from preprocessing."""

    original_bgr: np.ndarray       # untouched, as loaded from disk (BGR, OpenCV convention)
    processed_bgr: np.ndarray      # resized + denoised, still 3-channel BGR
    processed_gray: np.ndarray     # single-channel grayscale, float-free uint8
    stats: PreprocessingStats
    source_path: Path


def validate_image_path(path: Path, cfg: PreprocessingConfig) -> None:
    """Validate that `path` points to a readable, supported image file.

    Raises ImageValidationError with a specific, human-readable reason on
    failure. This is called BEFORE attempting to load pixel data, so a
    missing file fails fast with a clear message instead of an OpenCV None
    downstream.
    """
    if not path.exists():
        raise ImageValidationError(f"Input path does not exist: {path}")

    if not path.is_file():
        raise ImageValidationError(f"Input path is not a file: {path}")

    if path.suffix.lower() not in cfg.allowed_extensions:
        raise ImageValidationError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Allowed: {cfg.allowed_extensions}"
        )

    if path.stat().st_size == 0:
        raise ImageValidationError(f"Input file is empty (0 bytes): {path}")


def load_image(path: Path, cfg: PreprocessingConfig) -> np.ndarray:
    """Load an image from disk as a BGR uint8 NumPy array.

    Uses cv2.imread, which returns None (rather than raising) on a corrupt
    or unreadable file -- we convert that into an explicit exception so
    callers never have to remember to check for None themselves.
    """
    validate_image_path(path, cfg)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(
            f"File could not be decoded as an image (possibly corrupted): {path}"
        )

    height, width = image.shape[:2]
    if min(height, width) < cfg.min_dimension:
        raise ImageValidationError(
            f"Image too small ({width}x{height}). "
            f"Minimum supported dimension is {cfg.min_dimension}px."
        )

    return image


def resize_if_needed(image: np.ndarray, cfg: PreprocessingConfig) -> tuple[np.ndarray, bool, float]:
    """Downscale `image` so its longer side <= cfg.max_dimension.

    Never upscales: upscaling would fabricate pixel detail that never
    existed, which is actively misleading for a forensics task (it could
    create spurious high-frequency artifacts that look like tampering
    evidence). Returns (resized_image, was_resized, scale_factor).
    """
    height, width = image.shape[:2]
    longer_side = max(height, width)

    if longer_side <= cfg.max_dimension:
        return image, False, 1.0

    scale_factor = cfg.max_dimension / float(longer_side)
    new_width = int(round(width * scale_factor))
    new_height = int(round(height * scale_factor))

    # INTER_AREA is the correct choice for shrinking images: it uses pixel
    # area relation, which avoids the aliasing that INTER_LINEAR/NEAREST
    # would introduce when downsampling -- aliasing would itself create
    # false high-frequency signal that could confuse later feature/edge
    # analysis.
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, True, scale_factor


def reduce_noise(image_bgr: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Apply Gaussian smoothing to suppress sensor/JPEG noise.

    MATH: Gaussian blur convolves the image with a 2D Gaussian kernel
        G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    This is a low-pass filter: it attenuates high spatial-frequency content
    (fine noise) while preserving low-frequency structure (edges of objects,
    large tampered regions). A small kernel (default 5x5) is used
    deliberately -- a large kernel would also smear out genuine tampering
    boundaries we want to detect later, defeating the purpose.
    """
    k = cfg.gaussian_kernel_size
    if k % 2 == 0:
        k += 1  # Gaussian kernels must have odd size; correct silently but log it.
        logger.warning("gaussian_kernel_size was even; adjusted to %d", k)

    return cv2.GaussianBlur(image_bgr, (k, k), sigmaX=cfg.gaussian_sigma)


def apply_clahe_if_configured(image_bgr: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Optionally apply CLAHE to the luminance channel only.

    CLAHE = Contrast Limited Adaptive Histogram Equalization. Standard
    histogram equalization redistributes intensities globally; CLAHE does
    it in small tiles (here 8x8) with a clip limit that caps how much any
    single tile can be stretched, avoiding noise amplification in flat
    regions. Applied only to the L channel of LAB color space so color
    (chrominance) information is untouched.
    """
    if not cfg.apply_clahe:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
    )
    l_equalized = clahe.apply(l_channel)

    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to single-channel grayscale using OpenCV's luminance
    weighting (Y = 0.299R + 0.587G + 0.114B, applied to the BGR order
    internally). Grayscale is required by SIFT and most segmentation
    methods used downstream.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def preprocess_image(path: Path, cfg: Optional[PreprocessingConfig] = None) -> PreprocessingResult:
    """End-to-end preprocessing entry point used by main.py.

    Pipeline: validate -> load -> (preserve original) -> resize -> denoise
    -> optional CLAHE -> grayscale -> compute stats.
    """
    if cfg is None:
        cfg = PreprocessingConfig()

    logger.info("Validating and loading image: %s", path)
    original_bgr = load_image(path, cfg)

    mean_before = float(np.mean(original_bgr))
    std_before = float(np.std(original_bgr))

    resized_bgr, was_resized, scale_factor = resize_if_needed(original_bgr, cfg)
    denoised_bgr = reduce_noise(resized_bgr, cfg)
    processed_bgr = apply_clahe_if_configured(denoised_bgr, cfg)
    processed_gray = to_grayscale(processed_bgr)

    mean_after = float(np.mean(processed_bgr))
    std_after = float(np.std(processed_bgr))

    stats = PreprocessingStats(
        original_width=original_bgr.shape[1],
        original_height=original_bgr.shape[0],
        original_channels=original_bgr.shape[2] if original_bgr.ndim == 3 else 1,
        processed_width=processed_bgr.shape[1],
        processed_height=processed_bgr.shape[0],
        was_resized=was_resized,
        resize_scale_factor=scale_factor,
        mean_intensity_before=mean_before,
        mean_intensity_after=mean_after,
        std_intensity_before=std_before,
        std_intensity_after=std_after,
    )

    logger.info(
        "Preprocessing complete: %dx%d -> %dx%d (resized=%s)",
        stats.original_width, stats.original_height,
        stats.processed_width, stats.processed_height,
        was_resized,
    )

    return PreprocessingResult(
        original_bgr=original_bgr,
        processed_bgr=processed_bgr,
        processed_gray=processed_gray,
        stats=stats,
        source_path=path,
    )
