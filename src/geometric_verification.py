"""
src/geometric_verification.py
================================
MODULE 4: Geometric Verification (RANSAC / Robust Verification merged)

RESPONSIBILITY
--------------
- compare suspicious regions/features
- use geometric relationships where appropriate
- use RANSAC for robust estimation where justified
- reject false matches/outliers
- calculate geometric consistency

WHY THIS MODULE EXISTS
------------------------
Feature matching (Module 2) can produce false-positive matches purely by
descriptor similarity -- two unrelated but visually similar patches
(e.g. two bricks in a wall) can match well without any real copy-move
relationship. Geometric verification adds a second, independent check:
if a region was genuinely copy-moved, ALL of its matched keypoint pairs
must be explained by a SINGLE consistent geometric transform (translation,
rotation, uniform scaling -- modeled here as an affine transform). Random
coincidental matches will NOT agree on one consistent transform.

SYLLABUS MAPPING
-----------------
Module 2: Homography, DLT (Direct Linear Transform), RANSAC, epipolar
geometry concepts (here applied within a single image rather than between
two camera views, but the estimation machinery is identical).

MATHEMATICAL INTUITION -- AFFINE TRANSFORM + RANSAC
------------------------------------------------------
An affine transform maps point (x, y) to (x', y') via:
    [x']   [a  b] [x]   [tx]
    [y'] = [c  d] [y] + [ty]
i.e. 6 unknown parameters (a, b, c, d, tx, ty), covering rotation, uniform/
non-uniform scaling, shear, and translation -- exactly the class of
transforms a copy-move attacker typically applies (paste, maybe rotate or
resize slightly).

RANSAC (RANdom SAmple Consensus) estimates this transform robustly in the
presence of outlier matches:
    1. Repeatedly sample a minimal random subset of matched point pairs
       (3 pairs, minimum needed to solve for 6 affine parameters).
    2. Fit the transform to that subset.
    3. Count "inliers": matches from the FULL set whose reprojection error
       under this transform is below a threshold.
    4. Keep the transform with the most inliers across many random trials.
    5. Optionally refit using all inliers for a final, refined transform.
This is exactly why RANSAC is essential here rather than a simple least-
squares fit: least squares would be dragged off by even a few outlier
matches, while RANSAC's random-sampling + consensus-counting approach
naturally ignores them as long as they are a minority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import GeometricConfig
from src.feature_matching import FeatureMatchingResult

logger = logging.getLogger(__name__)


@dataclass
class GeometricVerificationResult:
    transform_found: bool
    affine_matrix: np.ndarray  # shape (2, 3), or None if not found
    num_inliers: int
    num_outliers: int
    inlier_ratio: float          # inliers / total matches considered, in [0, 1]
    inlier_query_pts: np.ndarray  # (N, 2)
    inlier_train_pts: np.ndarray  # (N, 2)
    geometric_consistency_score: float  # in [0, 1], fused into final tampering score

    def to_dict(self) -> dict:
        return {
            "transform_found": self.transform_found,
            "num_inliers": self.num_inliers,
            "num_outliers": self.num_outliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "geometric_consistency_score": round(self.geometric_consistency_score, 4),
        }


def verify_geometric_consistency(
    matching_result: FeatureMatchingResult,
    cfg: GeometricConfig,
) -> GeometricVerificationResult:
    """Fit a robust affine transform to the matched keypoint pairs via
    RANSAC, and report inlier/outlier statistics.

    Uses cv2.estimateAffinePartial2D (similarity transform: rotation +
    uniform scale + translation) rather than a full 6-DOF affine, because
    a similarity transform is the more realistic model for a copy-paste
    operation (attackers rotate/scale, but non-uniform shear is rare and
    would visually distort the pasted content in an obvious way). It still
    uses the same RANSAC machinery described in the module docstring.
    """
    n_matches = len(matching_result.match_pairs)

    if n_matches < cfg.min_inliers_for_verification:
        logger.info(
            "Only %d candidate matches (< %d required); skipping RANSAC.",
            n_matches, cfg.min_inliers_for_verification,
        )
        return GeometricVerificationResult(
            transform_found=False,
            affine_matrix=None,
            num_inliers=0,
            num_outliers=n_matches,
            inlier_ratio=0.0,
            inlier_query_pts=np.zeros((0, 2)),
            inlier_train_pts=np.zeros((0, 2)),
            geometric_consistency_score=0.0,
        )

    query_pts = np.array([p.query_pt for p in matching_result.match_pairs], dtype=np.float32)
    train_pts = np.array([p.train_pt for p in matching_result.match_pairs], dtype=np.float32)

    cv2.setRNGSeed(cfg.random_seed)

    affine_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        query_pts,
        train_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reproj_threshold,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )

    if affine_matrix is None or inlier_mask is None:
        logger.warning("RANSAC failed to find a consistent affine transform.")
        return GeometricVerificationResult(
            transform_found=False,
            affine_matrix=None,
            num_inliers=0,
            num_outliers=n_matches,
            inlier_ratio=0.0,
            inlier_query_pts=np.zeros((0, 2)),
            inlier_train_pts=np.zeros((0, 2)),
            geometric_consistency_score=0.0,
        )

    inlier_mask = inlier_mask.flatten().astype(bool)
    num_inliers = int(inlier_mask.sum())
    num_outliers = n_matches - num_inliers
    inlier_ratio = num_inliers / n_matches if n_matches > 0 else 0.0

    meets_minimum = num_inliers >= cfg.min_inliers_for_verification

    # Geometric consistency score: inlier ratio, but zeroed out if the
    # absolute inlier count doesn't clear the minimum -- a 100% inlier
    # ratio from only 2 matches is not meaningful evidence.
    geometric_score = inlier_ratio if meets_minimum else 0.0

    logger.info(
        "RANSAC: %d/%d matches are inliers (ratio=%.4f), transform_found=%s",
        num_inliers, n_matches, inlier_ratio, meets_minimum,
    )

    return GeometricVerificationResult(
        transform_found=meets_minimum,
        affine_matrix=affine_matrix,
        num_inliers=num_inliers,
        num_outliers=num_outliers,
        inlier_ratio=inlier_ratio,
        inlier_query_pts=query_pts[inlier_mask],
        inlier_train_pts=train_pts[inlier_mask],
        geometric_consistency_score=float(geometric_score),
    )
