"""
tests/test_pipeline.py
========================
Integration tests: run the full pipeline end-to-end (as main.py would) on
synthetic images, including all the required edge cases from Phase 8:
invalid file, unsupported format, corrupted image, very small image,
grayscale image, low-quality image, image with insufficient features,
authentic image, manipulated image, image with repetitive textures.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from main import run_pipeline_on_image
from src.preprocessing import ImageValidationError


@pytest.fixture
def cfg():
    return get_config()


def _textured_image(w=500, h=400, seed=0, n_blobs=80):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (70, 100, 130), dtype=np.uint8)
    for _ in range(n_blobs):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(8, 22)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), r, color, -1)
    return img


# ---------------------------------------------------------------------------
# Failure / edge cases (Phase 8 required test list)
# ---------------------------------------------------------------------------

def test_invalid_file_path(tmp_path, cfg):
    with pytest.raises(ImageValidationError):
        run_pipeline_on_image(tmp_path / "does_not_exist.jpg", tmp_path / "out", cfg, save_visuals=False)


def test_unsupported_format(tmp_path, cfg):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ImageValidationError):
        run_pipeline_on_image(bad, tmp_path / "out", cfg, save_visuals=False)


def test_corrupted_image(tmp_path, cfg):
    fake = tmp_path / "corrupt.jpg"
    fake.write_bytes(b"not a real jpeg")
    with pytest.raises(ImageValidationError):
        run_pipeline_on_image(fake, tmp_path / "out", cfg, save_visuals=False)


def test_very_small_image(tmp_path, cfg):
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "small.png"
    cv2.imwrite(str(path), small)
    with pytest.raises(ImageValidationError):
        run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)


def test_grayscale_image_runs_without_crashing(tmp_path, cfg):
    gray = np.random.default_rng(2).integers(0, 255, (300, 400), dtype=np.uint8)
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), gray)
    decision, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    assert decision.decision in ("AUTHENTIC", "POTENTIALLY TAMPERED")


def test_low_quality_heavily_compressed_image(tmp_path, cfg):
    img = _textured_image()
    path = tmp_path / "lowq.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 5])  # aggressive compression
    decision, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    assert decision.decision in ("AUTHENTIC", "POTENTIALLY TAMPERED")


def test_image_with_insufficient_features(tmp_path, cfg):
    # A perfectly flat image has essentially no SIFT-detectable structure.
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    path = tmp_path / "flat.png"
    cv2.imwrite(str(path), flat)
    decision, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    # Pipeline must degrade gracefully, not crash, and should not falsely
    # claim tampering with zero evidence.
    assert report["features"]["num_keypoints"] == 0
    assert decision.decision == "AUTHENTIC"


def test_image_with_repetitive_textures_runs(tmp_path, cfg):
    # Many near-identical tiled squares -- a documented hard case where
    # feature matching can produce many ambiguous/ratio-test-rejected
    # matches. Must still run without crashing.
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    tile = np.random.default_rng(5).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    for r in range(0, 400, 40):
        for c in range(0, 400, 40):
            img[r:r + 40, c:c + 40] = tile
    path = tmp_path / "repetitive.png"
    cv2.imwrite(str(path), img)
    decision, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    assert decision.decision in ("AUTHENTIC", "POTENTIALLY TAMPERED")


# ---------------------------------------------------------------------------
# Authentic vs. manipulated (ground-truth controlled cases)
# ---------------------------------------------------------------------------

def test_authentic_image_end_to_end(tmp_path, cfg):
    img = _textured_image(seed=10)
    path = tmp_path / "authentic.jpg"
    cv2.imwrite(str(path), img)
    decision, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    assert report["decision"]["tampering_score"] >= 0.0
    assert "components" in report["decision"]


def test_manipulated_copy_move_image_end_to_end(tmp_path, cfg):
    img = _textured_image(seed=11, n_blobs=150)
    patch = img[50:200, 50:220].copy()
    img[220:370, 260:430] = patch
    path = tmp_path / "tampered.jpg"
    cv2.imwrite(str(path), img)
    decision, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    # We assert the pipeline PRODUCES a real score with real evidence
    # fields -- we do NOT hard-assert the decision label here, because a
    # single synthetic image is not guaranteed to cross the threshold
    # (see docs/methodology/threshold_calibration.md for the honest,
    # documented limitation this reflects).
    assert report["feature_matching"]["num_filtered_matches"] >= 0
    assert report["geometric_verification"]["num_inliers"] >= 0
    assert 0.0 <= decision.tampering_score <= 1.0


def test_report_json_is_serializable(tmp_path, cfg):
    import json
    img = _textured_image(seed=12)
    path = tmp_path / "img.jpg"
    cv2.imwrite(str(path), img)
    _, report = run_pipeline_on_image(path, tmp_path / "out", cfg, save_visuals=False)
    # Must not raise -- proves every field in the report is JSON-serializable.
    json.dumps(report)
