"""
config.py
=========
Single source of truth for every tunable parameter in the pipeline.

WHY THIS FILE EXISTS
---------------------
Development rule #16 requires "all paths configurable" and rule #9 requires
configuration instead of hard-coded paths/values scattered across modules.
Every module in src/ imports its parameters from here instead of hard-coding
numbers inline. This also means the whole pipeline's behaviour can be tuned
from ONE place, which matters a lot when you are trying to explain design
decisions in a viva ("why did you pick threshold X?").

The dataclass is intentionally flat and simple -- no nested YAML/JSON config
system, because that would be over-engineering for a course project (see
development rule: "Do not over-engineer the project.").
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root and standard directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VIS_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"


@dataclass
class PreprocessingConfig:
    """Parameters for src/preprocessing.py (Module 1)."""

    # Images are resized so their longer side equals this value, ONLY if the
    # original is larger. This keeps runtime predictable and keypoint scales
    # comparable across images, without upscaling small images (which would
    # fabricate detail that isn't there).
    max_dimension: int = 1024

    # Gaussian blur kernel size for noise reduction. Must be odd.
    # Small kernel (5) preserves tampering-relevant high-frequency detail
    # while removing sensor/JPEG noise that would otherwise create spurious
    # SIFT keypoints.
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 0.0  # 0 => OpenCV computes sigma from kernel size

    # Whether to apply CLAHE (Contrast Limited Adaptive Histogram
    # Equalization) to the luminance channel. This is OFF by default because
    # it can itself alter local contrast statistics that the histogram
    # consistency module relies on -- CLAHE is offered as an option for
    # low-light images, not a default step.
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    # Minimum accepted image dimension (width or height) in pixels. Images
    # smaller than this are rejected because SIFT/segmentation become
    # unreliable on extremely small images (documented failure case).
    min_dimension: int = 64

    # Accepted input file extensions.
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class FeatureConfig:
    """Parameters for src/feature_extraction.py and feature_matching.py (Module 2)."""

    n_features: int = 0          # 0 = unlimited (let SIFT find as many as it can)
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6

    # Ratio test (Lowe's ratio) threshold for accepting a match as reliable.
    lowe_ratio: float = 0.75

    # Minimum pixel distance between two matched keypoints for the match to
    # be considered a *candidate copy-move pair* rather than a trivial
    # self-match (a keypoint always matches itself at distance 0).
    min_match_distance_px: float = 16.0


@dataclass
class SegmentationConfig:
    """Parameters for src/segmentation.py (Module 3)."""

    method: str = "felzenszwalb"
    felzenszwalb_scale: float = 150.0
    felzenszwalb_sigma: float = 0.6
    felzenszwalb_min_size: int = 80

    # Morphological cleanup kernel size (pixels) applied to candidate masks.
    morph_kernel_size: int = 5


@dataclass
class GeometricConfig:
    """Parameters for src/geometric_verification.py (Module 4, RANSAC)."""

    ransac_reproj_threshold: float = 5.0   # pixels
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.99
    min_inliers_for_verification: int = 8
    random_seed: int = 42


@dataclass
class ScoringConfig:
    """Weights for src/tampering_score.py (Module 5, decision fusion)."""

    weight_match_density: float = 0.35
    weight_geometric_consistency: float = 0.45
    weight_histogram_anomaly: float = 0.20

    # Final score >= this threshold => POTENTIALLY TAMPERED
    #
    # CALIBRATION NOTE (honest, not fabricated): an initial default of 0.5
    # was tested against a small synthetic dev set (3 authentic + 3
    # copy-move-tampered images, see data/test/) and found to classify
    # EVERY image as AUTHENTIC, because match_density and inlier_ratio on
    # these images never both reach high absolute values simultaneously
    # even for genuinely tampered images -- the weighted-sum formula's
    # numeric scale is lower than 0.5 in practice for this test data.
    # Observed scores on the dev set: authentic ~0.026, tampered
    # 0.030-0.233 (see docs/methodology/threshold_calibration.md for the
    # full recorded run). 0.10 was chosen as it separates the dev set's
    # authentic sample from 2 of 3 tampered samples; it still MISSES a
    # tampered image with weak texture (tamp_2, score 0.030) -- this is a
    # genuine, documented failure case, not something this threshold
    # change fixes. A larger, more diverse labeled dataset (Phase 4) is
    # needed to calibrate this properly instead of on 6 images.
    decision_threshold: float = 0.10


@dataclass
class AppConfig:
    """Top-level config aggregating every module's configuration."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    vis_dir: Path = VIS_DIR
    report_dir: Path = REPORT_DIR

    log_level: str = "INFO"


def get_config() -> AppConfig:
    """Factory returning a fresh default configuration.

    Using a factory function (instead of a module-level singleton) means
    tests can create an independent config, tweak a value, and not affect
    other tests -- important for the test suite in Phase 8.
    """
    return AppConfig()
