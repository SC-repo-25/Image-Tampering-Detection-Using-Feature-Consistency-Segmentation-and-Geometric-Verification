"""
tests/test_segmentation.py
============================
Unit tests for src/segmentation.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SegmentationConfig
from src.segmentation import (
    build_candidate_mask,
    identify_candidate_segments,
    segment_and_localize,
    segment_image,
)


@pytest.fixture
def cfg():
    return SegmentationConfig()


def _blocky_image(w=300, h=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 3] = (30, 30, 30)
    img[:, w // 3: 2 * w // 3] = (150, 150, 150)
    img[:, 2 * w // 3:] = (230, 230, 230)
    return img


def test_segment_image_produces_label_map(cfg):
    img = _blocky_image()
    seg_map = segment_image(img, cfg)
    assert seg_map.shape == (200, 300)
    assert seg_map.max() >= 0
    # A 3-color-block image should segment into at least a few distinct regions.
    assert len(np.unique(seg_map)) >= 2


def test_identify_candidate_segments_empty_pairs_yields_empty(cfg):
    img = _blocky_image()
    seg_map = segment_image(img, cfg)
    candidates = identify_candidate_segments(seg_map, [])
    assert candidates == []


def test_identify_candidate_segments_finds_correct_segment(cfg):
    img = _blocky_image()
    seg_map = segment_image(img, cfg)
    # A point pair both landing in the leftmost dark block.
    pairs = [((10.0, 10.0), (20.0, 20.0))]
    candidates = identify_candidate_segments(seg_map, pairs)
    assert len(candidates) >= 1


def test_build_candidate_mask_shape_and_binary(cfg):
    img = _blocky_image()
    seg_map = segment_image(img, cfg)
    candidates = identify_candidate_segments(seg_map, [((10.0, 10.0), (20.0, 20.0))])
    mask = build_candidate_mask(seg_map, candidates, cfg)
    assert mask.shape == seg_map.shape
    assert set(np.unique(mask)).issubset({0, 255})


def test_segment_and_localize_no_matches_gives_empty_mask(cfg):
    img = _blocky_image()
    result = segment_and_localize(img, [], cfg)
    assert result.candidate_segment_ids == []
    assert np.count_nonzero(result.candidate_mask) == 0


def test_segment_and_localize_with_matches_flags_region(cfg):
    img = _blocky_image()
    pairs = [((10.0, 10.0), (150.0, 100.0))]
    result = segment_and_localize(img, pairs, cfg)
    assert len(result.candidate_segment_ids) >= 1
    assert np.count_nonzero(result.candidate_mask) > 0
