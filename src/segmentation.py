"""
src/segmentation.py
=====================
MODULE 3: Segmentation & Candidate Tampered Region Detection

RESPONSIBILITY
--------------
- identify suspicious regions
- use an appropriate segmentation method
- generate a binary/multi-region mask
- perform morphological cleanup
- calculate candidate region statistics

APPROACH
--------
1. Segment the whole image into superpixel-like regions using Felzenszwalb's
   graph-based segmentation (skimage) -- this groups pixels into perceptually
   coherent regions based on color/texture similarity, without needing any
   training data.
2. Use the matched keypoint pairs from feature_matching.py to identify WHICH
   segments contain matched keypoints -- these segments become the candidate
   suspicious regions (both the "source" and the "copy" side of a possible
   copy-move pair).
3. Build a binary mask from those candidate segments and clean it up
   morphologically (closing small gaps, removing tiny noise blobs).

SYLLABUS MAPPING
-----------------
Module 3: Region Growing, Edge-based segmentation, Graph Cut, Mean Shift,
texture segmentation. Felzenszwalb is a graph-based segmentation method in
the same family as Graph Cut (both build a graph over pixels and cut/merge
based on edge weights), which is explicitly why it was chosen over K-Means
color clustering (Module 4) -- K-Means ignores spatial/graph structure and
tends to produce far noisier region boundaries for this purpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb

from config import SegmentationConfig
from src.feature_matching import FeatureMatchingResult

logger = logging.getLogger(__name__)

# NOTE ON PIPELINE ORDERING (see main.py):
# identify_candidate_segments() below accepts either the raw
# FeatureMatchingResult or, preferably, a pre-filtered list of (pt1, pt2)
# tuples restricted to RANSAC inliers. Building the localization mask from
# RAW (pre-RANSAC) matches over-flags the image, because self-matching
# alone cannot distinguish "genuinely copy-moved region" from "two
# unrelated but visually similar patches" (e.g. two similar-looking circles
# in a synthetic/repetitive-texture image) -- exactly the failure mode
# RANSAC geometric verification exists to reject. main.py therefore runs
# segmentation AFTER geometric verification and passes inlier-only point
# pairs, even though segmentation is architecturally "Module 3" and
# geometric verification is "Module 4" -- the localization mask specifically
# is computed from Module 4's output, while whole-image segmentation itself
# still only needs the image.


@dataclass
class SegmentStats:
    segment_id: int
    pixel_count: int
    bounding_box: tuple  # (x, y, w, h)
    is_candidate: bool


@dataclass
class SegmentationResult:
    segment_map: np.ndarray          # int array, same H x W, each pixel labeled with segment id
    num_segments: int
    candidate_mask: np.ndarray       # uint8 binary mask, 255 where suspicious
    candidate_segment_ids: list
    segment_stats: list              # list[SegmentStats]

    def to_dict(self) -> dict:
        return {
            "num_segments": self.num_segments,
            "num_candidate_segments": len(self.candidate_segment_ids),
            "candidate_pixel_fraction": round(
                float(np.count_nonzero(self.candidate_mask)) / self.candidate_mask.size, 4
            ) if self.candidate_mask.size else 0.0,
        }


def segment_image(image_bgr: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Run Felzenszwalb graph-based segmentation.

    MATH/ALGORITHM INTUITION: the image is modeled as a graph where each
    pixel is a node and edges connect neighboring pixels, weighted by color
    difference. The algorithm greedily merges regions if the difference
    between them is small relative to the internal variation already present
    within each region (an adaptive, region-specific threshold rather than a
    single global one) -- this lets it segment both smooth and textured
    areas sensibly in one pass.

    Params (from cfg):
        scale: larger => larger, fewer segments (controls the merging
            threshold sensitivity).
        sigma: Gaussian smoothing applied before segmentation (reduces
            noise-driven over-segmentation).
        min_size: minimum segment size in pixels; smaller segments are
            merged into a neighbor.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    segment_map = felzenszwalb(
        rgb,
        scale=cfg.felzenszwalb_scale,
        sigma=cfg.felzenszwalb_sigma,
        min_size=cfg.felzenszwalb_min_size,
    )
    return segment_map.astype(np.int32)


def identify_candidate_segments(
    segment_map: np.ndarray,
    point_pairs,
) -> list:
    """Return the set of segment IDs that contain at least one matched
    keypoint from `point_pairs`.

    Args:
        point_pairs: an iterable of ((x1, y1), (x2, y2)) tuples. Callers
            should pass RANSAC-INLIER point pairs (from
            GeometricVerificationResult.inlier_query_pts /
            inlier_train_pts) rather than raw, pre-RANSAC matches, so that
            localization reflects geometrically-verified evidence only --
            see the module-level note above for why this matters.
    """
    candidate_ids = set()
    height, width = segment_map.shape

    for pt_a, pt_b in point_pairs:
        for (x, y) in (pt_a, pt_b):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                candidate_ids.add(int(segment_map[yi, xi]))

    return sorted(candidate_ids)


def build_candidate_mask(
    segment_map: np.ndarray,
    candidate_ids: list,
    cfg: SegmentationConfig,
) -> np.ndarray:
    """Build a binary mask marking pixels belonging to candidate segments,
    then apply morphological cleanup.

    MORPHOLOGICAL CLEANUP:
    - Closing (dilation followed by erosion) fills small holes/gaps inside
      candidate regions, so a suspicious region isn't reported as several
      disconnected fragments.
    - Opening (erosion followed by dilation) removes isolated single-pixel
      noise blobs that don't represent a meaningful region.
    """
    mask = np.isin(segment_map, candidate_ids).astype(np.uint8) * 255

    k = cfg.morph_kernel_size
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def compute_segment_stats(
    segment_map: np.ndarray,
    candidate_ids: list,
) -> list:
    """Compute per-segment pixel count and bounding box for reporting."""
    stats = []
    unique_ids = np.unique(segment_map)

    for seg_id in unique_ids:
        ys, xs = np.where(segment_map == seg_id)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        stats.append(
            SegmentStats(
                segment_id=int(seg_id),
                pixel_count=int(len(xs)),
                bounding_box=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                is_candidate=int(seg_id) in candidate_ids,
            )
        )
    return stats


def segment_and_localize(
    image_bgr: np.ndarray,
    point_pairs,
    cfg: SegmentationConfig,
) -> SegmentationResult:
    """End-to-end entry point used by main.py.

    Args:
        point_pairs: RANSAC-inlier point pairs -- see
            identify_candidate_segments() docstring. Pass an empty list if
            geometric verification found no consistent transform, which
            correctly yields an empty candidate mask (nothing localized)
            rather than over-flagging the whole image.
    """
    segment_map = segment_image(image_bgr, cfg)
    num_segments = int(segment_map.max()) + 1

    candidate_ids = identify_candidate_segments(segment_map, point_pairs)
    candidate_mask = build_candidate_mask(segment_map, candidate_ids, cfg)
    segment_stats = compute_segment_stats(segment_map, candidate_ids)

    logger.info(
        "Segmentation: %d total segments, %d flagged as candidates",
        num_segments, len(candidate_ids),
    )

    return SegmentationResult(
        segment_map=segment_map,
        num_segments=num_segments,
        candidate_mask=candidate_mask,
        candidate_segment_ids=candidate_ids,
        segment_stats=segment_stats,
    )
