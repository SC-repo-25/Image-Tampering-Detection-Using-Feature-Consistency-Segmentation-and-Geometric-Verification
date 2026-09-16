"""
src/feature_matching.py
=========================
PART OF MODULE 2: Image Feature & Consistency Analysis

RESPONSIBILITY
--------------
Match SIFT descriptors AGAINST THEMSELVES (the image's own keypoint set)
to find pairs of keypoints that describe near-identical local patches at
DIFFERENT spatial locations. If a region was copy-moved within the image,
its duplicated keypoints will match this way. This is the classic
"self-correlation" approach to copy-move forgery detection.

SYLLABUS MAPPING
-----------------
Module 3: feature descriptors, orientation histograms (SIFT matching).

KEY DESIGN DECISION: excluding trivial self-matches
-----------------------------------------------------
Every keypoint is a perfect match for itself (distance 0). We must exclude
these before applying Lowe's ratio test, and additionally reject matches
whose two keypoints are close together in image space (min_match_distance_px)
since neighbouring, overlapping keypoints on the same real structure
naturally have similar descriptors -- that is NOT evidence of tampering,
just an artifact of dense keypoint sampling on textured surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import FeatureConfig
from src.feature_extraction import FeatureExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class MatchPair:
    query_idx: int
    train_idx: int
    query_pt: tuple    # (x, y)
    train_pt: tuple    # (x, y)
    distance: float    # descriptor distance (lower = more similar)
    spatial_distance: float  # pixel distance between the two keypoints


@dataclass
class FeatureMatchingResult:
    match_pairs: list  # list[MatchPair], self-matches only, ratio-tested
    num_raw_matches: int
    num_filtered_matches: int
    match_density: float  # filtered matches / total keypoints, in [0, 1]

    def to_dict(self) -> dict:
        return {
            "num_raw_matches": self.num_raw_matches,
            "num_filtered_matches": self.num_filtered_matches,
            "match_density": round(self.match_density, 4),
        }


def match_features_self(
    features: FeatureExtractionResult,
    cfg: FeatureConfig,
) -> FeatureMatchingResult:
    """Find candidate copy-move keypoint pairs within a single image.

    ALGORITHM:
    1. Build a k-NN matcher (k=3, since k=1 would always be the keypoint
       itself at distance 0; we need the top few OTHER matches).
    2. For each keypoint, skip the self-match (distance ~0, same index),
       then apply Lowe's ratio test between the best remaining match and
       the second-best: a match is accepted only if
           best_distance < lowe_ratio * second_best_distance
       This rejects ambiguous matches where many keypoints look similar
       (e.g. repetitive texture), keeping only distinctive, reliable pairs
       -- exactly the failure case ("image with repetitive textures")
       this project is required to test against.
    3. Reject matches whose two keypoints are spatially closer than
       `min_match_distance_px` (see module docstring).
    4. De-duplicate symmetric pairs (i matches j and j matches i counted once).
    """
    n = features.descriptors.shape[0]
    if n < 4:
        logger.warning("Too few keypoints (%d) for self-matching.", n)
        return FeatureMatchingResult(
            match_pairs=[], num_raw_matches=0, num_filtered_matches=0, match_density=0.0
        )

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # k=3: index 0 will always be the trivial self-match; 1 and 2 are the
    # best and second-best genuine candidates.
    knn_matches = bf.knnMatch(features.descriptors, features.descriptors, k=3)

    seen_pairs = set()
    accepted = []
    raw_count = 0

    for matches in knn_matches:
        if len(matches) < 3:
            continue

        # matches[0] is the keypoint matching itself (distance == 0);
        # skip it and evaluate matches[1] vs matches[2] with the ratio test.
        best, second = matches[1], matches[2]
        raw_count += 1

        if second.distance <= 1e-10:
            continue

        if best.distance >= cfg.lowe_ratio * second.distance:
            continue  # ambiguous match, reject

        q_idx, t_idx = best.queryIdx, best.trainIdx
        if q_idx == t_idx:
            continue

        q_pt = features.keypoint_coords[q_idx]
        t_pt = features.keypoint_coords[t_idx]
        spatial_dist = float(np.linalg.norm(q_pt - t_pt))

        if spatial_dist < cfg.min_match_distance_px:
            continue  # too close -- likely dense sampling on one real structure

        pair_key = tuple(sorted((q_idx, t_idx)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        accepted.append(
            MatchPair(
                query_idx=q_idx,
                train_idx=t_idx,
                query_pt=(float(q_pt[0]), float(q_pt[1])),
                train_pt=(float(t_pt[0]), float(t_pt[1])),
                distance=float(best.distance),
                spatial_distance=spatial_dist,
            )
        )

    match_density = len(accepted) / n if n > 0 else 0.0

    logger.info(
        "Self-matching: %d raw candidates -> %d filtered pairs (density=%.4f)",
        raw_count, len(accepted), match_density,
    )

    return FeatureMatchingResult(
        match_pairs=accepted,
        num_raw_matches=raw_count,
        num_filtered_matches=len(accepted),
        match_density=match_density,
    )
