"""
tests/test_features.py
========================
Unit tests for src/feature_extraction.py and src/feature_matching.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FeatureConfig
from src.feature_extraction import extract_features, extract_sift_features
from src.feature_matching import match_features_self


@pytest.fixture
def cfg():
    return FeatureConfig()


def _textured_gray(w=400, h=300, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (70, 100, 130), dtype=np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(6, 20)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), r, color, -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_sift_finds_keypoints_on_textured_image(cfg):
    gray = _textured_gray()
    result = extract_sift_features(gray, cfg)
    assert len(result.keypoints) > 0
    assert result.descriptors.shape[0] == len(result.keypoints)
    assert result.descriptors.shape[1] == 128
    assert result.detector_name == "SIFT"


def test_sift_handles_blank_image_gracefully(cfg):
    blank = np.full((200, 200), 128, dtype=np.uint8)  # flat, no texture at all
    result = extract_sift_features(blank, cfg)
    # Must not raise; descriptors array must have correct shape even when empty.
    assert result.descriptors.shape[1] == 128
    assert len(result.keypoints) == result.descriptors.shape[0]


def test_extract_features_dispatch_orb(cfg):
    gray = _textured_gray()
    result = extract_features(gray, cfg, use_orb=True)
    assert result.detector_name == "ORB"
    assert result.descriptors.shape[0] == len(result.keypoints)


def test_self_matching_on_duplicated_region(cfg):
    # Build an image containing a duplicated patch -- a controlled,
    # ground-truth copy-move case -- and confirm self-matching finds
    # correspondences between the two copies.
    rng = np.random.default_rng(1)
    img = np.full((400, 500, 3), (70, 100, 130), dtype=np.uint8)
    for _ in range(80):
        x, y = rng.integers(0, 500), rng.integers(0, 400)
        r = rng.integers(6, 18)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), r, color, -1)

    patch = img[40:160, 40:200].copy()
    img[220:340, 280:440] = patch  # paste far away, no overlap

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = extract_sift_features(gray, cfg)
    matching = match_features_self(features, cfg)

    # We can't guarantee an exact count (depends on how many keypoints the
    # patch happened to contain), but a duplicated region with real texture
    # should yield at least one filtered match pair.
    assert matching.num_filtered_matches >= 1
    assert matching.match_density >= 0.0


def test_self_matching_rejects_nearby_trivial_matches(cfg):
    gray = _textured_gray()
    features = extract_sift_features(gray, cfg)
    matching = match_features_self(features, cfg)

    for pair in matching.match_pairs:
        assert pair.spatial_distance >= cfg.min_match_distance_px


def test_self_matching_too_few_keypoints(cfg):
    tiny = np.full((30, 30), 128, dtype=np.uint8)
    features = extract_sift_features(tiny, cfg)
    matching = match_features_self(features, cfg)
    assert matching.num_filtered_matches == 0
    assert matching.match_density == 0.0
