"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py (Module 1).

These tests generate small synthetic images in-memory (no dataset needed)
so the module can be verified in isolation, before any real dataset is
wired up. Run with:

    pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PreprocessingConfig
from src.preprocessing import (
    ImageValidationError,
    load_image,
    preprocess_image,
    reduce_noise,
    resize_if_needed,
    to_grayscale,
    validate_image_path,
)


@pytest.fixture
def cfg() -> PreprocessingConfig:
    return PreprocessingConfig()


@pytest.fixture
def tmp_image_factory(tmp_path):
    """Returns a function that writes a synthetic test image to disk and
    returns its path. Using a factory fixture lets each test create images
    with different sizes/content.
    """

    def _make(filename: str = "test.png", width: int = 300, height: int = 200,
              color: bool = True) -> Path:
        if color:
            # Deterministic synthetic pattern, not random noise, so tests
            # are reproducible.
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, : width // 2] = (40, 80, 120)   # BGR block 1
            img[:, width // 2:] = (200, 180, 60)   # BGR block 2
            cv2.rectangle(img, (10, 10), (60, 60), (255, 255, 255), -1)
        else:
            img = np.random.default_rng(0).integers(0, 255, (height, width), dtype=np.uint8)

        path = tmp_path / filename
        cv2.imwrite(str(path), img)
        return path

    return _make


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_missing_file_raises(cfg):
    with pytest.raises(ImageValidationError, match="does not exist"):
        validate_image_path(Path("/nonexistent/path/image.jpg"), cfg)


def test_validate_unsupported_extension_raises(tmp_path, cfg):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not an image")
    with pytest.raises(ImageValidationError, match="Unsupported file extension"):
        validate_image_path(bad_file, cfg)


def test_validate_empty_file_raises(tmp_path, cfg):
    empty_file = tmp_path / "empty.jpg"
    empty_file.touch()
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image_path(empty_file, cfg)


def test_load_corrupted_file_raises(tmp_path, cfg):
    fake_jpg = tmp_path / "corrupt.jpg"
    fake_jpg.write_bytes(b"this is not actually image data at all")
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        load_image(fake_jpg, cfg)


def test_load_too_small_image_raises(tmp_image_factory, cfg):
    path = tmp_image_factory(width=20, height=20)
    with pytest.raises(ImageValidationError, match="too small"):
        load_image(path, cfg)


def test_load_valid_image_succeeds(tmp_image_factory, cfg):
    path = tmp_image_factory(width=300, height=200)
    img = load_image(path, cfg)
    assert img.shape == (200, 300, 3)
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Resize tests
# ---------------------------------------------------------------------------

def test_resize_skips_small_images(cfg):
    small = np.zeros((100, 150, 3), dtype=np.uint8)
    resized, was_resized, scale = resize_if_needed(small, cfg)
    assert was_resized is False
    assert scale == 1.0
    assert resized.shape == small.shape


def test_resize_shrinks_large_images(cfg):
    large = np.zeros((2000, 3000, 3), dtype=np.uint8)
    resized, was_resized, scale = resize_if_needed(large, cfg)
    assert was_resized is True
    assert max(resized.shape[:2]) == cfg.max_dimension
    assert scale < 1.0


def test_resize_never_upscales(cfg):
    small = np.zeros((50, 60, 3), dtype=np.uint8)
    # Even with a tiny max_dimension below the image size on one axis only,
    # resize should never make the image LARGER than original.
    resized, _, _ = resize_if_needed(small, cfg)
    assert resized.shape[0] <= 50 or resized.shape[0] == small.shape[0]


# ---------------------------------------------------------------------------
# Noise reduction / grayscale tests
# ---------------------------------------------------------------------------

def test_reduce_noise_preserves_shape_and_dtype(cfg):
    img = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    blurred = reduce_noise(img, cfg)
    assert blurred.shape == img.shape
    assert blurred.dtype == img.dtype


def test_reduce_noise_actually_smooths(cfg):
    # Salt-and-pepper-like extreme noise pattern; after blurring, the
    # standard deviation of intensities should decrease (smoothing effect).
    rng = np.random.default_rng(2)
    noisy = rng.choice([0, 255], size=(100, 100, 3)).astype(np.uint8)
    blurred = reduce_noise(noisy, cfg)
    assert np.std(blurred) < np.std(noisy)


def test_to_grayscale_reduces_channels():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    gray = to_grayscale(img)
    assert gray.ndim == 2
    assert gray.shape == (50, 50)


def test_even_kernel_size_is_corrected(cfg):
    cfg.gaussian_kernel_size = 4  # invalid, even
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # Should not raise -- reduce_noise auto-corrects to an odd kernel.
    result = reduce_noise(img, cfg)
    assert result.shape == img.shape


# ---------------------------------------------------------------------------
# End-to-end preprocess_image tests
# ---------------------------------------------------------------------------

def test_preprocess_image_end_to_end(tmp_image_factory, cfg):
    path = tmp_image_factory(width=1500, height=1000)
    result = preprocess_image(path, cfg)

    assert result.original_bgr.shape == (1000, 1500, 3)
    assert max(result.processed_bgr.shape[:2]) == cfg.max_dimension
    assert result.processed_gray.ndim == 2
    assert result.stats.was_resized is True
    assert result.stats.original_width == 1500
    assert result.stats.original_height == 1000
    # Original must remain untouched (module must not mutate input).
    assert result.original_bgr.shape[1] == 1500


def test_preprocess_image_original_not_mutated_by_resize(tmp_image_factory, cfg):
    path = tmp_image_factory(width=1500, height=1000)
    result = preprocess_image(path, cfg)
    # original_bgr should be a different array (or at least different shape)
    # from processed_bgr, proving no in-place resize happened on the source.
    assert result.original_bgr.shape != result.processed_bgr.shape


def test_preprocess_image_grayscale_input_file(tmp_path, cfg):
    # A single-channel PNG saved to disk; cv2.imread with IMREAD_COLOR will
    # still load it as 3-channel BGR (replicated channel), which is the
    # correct, documented OpenCV behaviour -- verify pipeline handles it.
    gray_img = np.random.default_rng(3).integers(0, 255, (150, 150), dtype=np.uint8)
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), gray_img)

    result = preprocess_image(path, cfg)
    assert result.original_bgr.shape == (150, 150, 3)
    assert result.processed_gray.shape == (150, 150)
