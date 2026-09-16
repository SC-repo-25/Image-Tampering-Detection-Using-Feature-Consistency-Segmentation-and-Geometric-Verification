"""
src/feature_extraction.py
===========================
PART OF MODULE 2: Image Feature & Consistency Analysis

RESPONSIBILITY
--------------
Extract scale- and rotation-invariant local keypoints and descriptors using
SIFT (Scale-Invariant Feature Transform). These features are the raw
material for detecting copy-move forgery: if a region was copied elsewhere
in the same image, its keypoints will have near-identical descriptors to
the keypoints in the original region.

SYLLABUS MAPPING
-----------------
Module 3: SIFT, scale-space analysis, image pyramids, orientation histogram.

WHY SIFT (not just Harris corners or ORB)
-------------------------------------------
- Harris corners (also in the syllabus) detect WHERE interesting points are
  but provide no descriptor for comparing them across regions -- we need
  both detection AND a matchable descriptor.
- SIFT builds a scale-space pyramid (Difference-of-Gaussians approximation
  of Laplacian-of-Gaussian) so keypoints are detected at the scale where
  they are most stable, then assigns each keypoint a dominant orientation
  and a 128-dim gradient-histogram descriptor invariant to that scale and
  rotation. This directly matters for copy-move detection, where the pasted
  region may be resized or rotated relative to its source.
- ORB is documented as a valid, faster alternative (see `use_orb=True`)
  but SIFT is used by default since it is the syllabus-named algorithm and
  gives more stable descriptors for this task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionResult:
    keypoints: list            # list[cv2.KeyPoint]
    descriptors: np.ndarray    # shape (n_keypoints, 128) for SIFT, float32
    keypoint_coords: np.ndarray  # shape (n_keypoints, 2), (x, y) for convenience
    detector_name: str

    def to_dict(self) -> dict:
        return {
            "detector": self.detector_name,
            "num_keypoints": len(self.keypoints),
        }


def extract_sift_features(
    gray_image: np.ndarray,
    cfg: FeatureConfig,
) -> FeatureExtractionResult:
    """Detect SIFT keypoints and compute their 128-dim descriptors.

    MATHEMATICAL INTUITION:
    1. Scale space: the image is progressively Gaussian-blurred at
       increasing sigma and downsampled into octaves, approximating how the
       same object looks at different viewing distances/resolutions.
    2. Difference-of-Gaussians (DoG): adjacent blurred images are
       subtracted; DoG approximates the (more expensive) Laplacian-of-
       Gaussian, whose extrema correspond to blob-like features that are
       stable across scale.
    3. Keypoint localization: local extrema of DoG across scale and space
       are candidate keypoints; low-contrast and edge-like candidates
       (unstable to noise) are discarded using `contrastThreshold` and
       `edgeThreshold`.
    4. Orientation assignment: a dominant gradient orientation is computed
       from a local gradient histogram, making the descriptor rotation
       invariant.
    5. Descriptor: a 128-dim vector from 4x4 spatial bins x 8 orientation
       bins of gradient histograms around the keypoint, normalized to
       reduce sensitivity to illumination change.

    Args:
        gray_image: single-channel uint8 image (output of preprocessing).
        cfg: FeatureConfig with detector parameters.

    Returns:
        FeatureExtractionResult with keypoints, descriptors, and coordinates.
    """
    sift = cv2.SIFT_create(
        nfeatures=cfg.n_features,
        contrastThreshold=cfg.contrast_threshold,
        edgeThreshold=cfg.edge_threshold,
        sigma=cfg.sigma,
    )

    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    if descriptors is None:
        logger.warning(
            "No SIFT keypoints found. Image may be too smooth/low-texture "
            "(e.g. blank wall, sky, or a very small/blurred image)."
        )
        descriptors = np.zeros((0, 128), dtype=np.float32)
        keypoint_coords = np.zeros((0, 2), dtype=np.float32)
    else:
        keypoint_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    logger.info("Extracted %d SIFT keypoints", len(keypoints) if keypoints else 0)

    return FeatureExtractionResult(
        keypoints=list(keypoints) if keypoints else [],
        descriptors=descriptors,
        keypoint_coords=keypoint_coords,
        detector_name="SIFT",
    )


def extract_orb_features(gray_image: np.ndarray, cfg: FeatureConfig) -> FeatureExtractionResult:
    """Alternative, faster feature detector (documented fallback, not default).

    ORB (Oriented FAST and Rotated BRIEF) combines the FAST corner detector
    with a rotation-aware binary BRIEF descriptor. It is ~an order of
    magnitude faster than SIFT and patent-unencumbered, making it a
    reasonable substitution if runtime on very large batches becomes a
    concern during --evaluate mode. Not used by default because binary
    Hamming-distance matching is coarser than SIFT's float descriptors for
    this project's needs, and SIFT is the syllabus-named algorithm.
    """
    n = cfg.n_features if cfg.n_features > 0 else 2000
    orb = cv2.ORB_create(nfeatures=n)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    if descriptors is None:
        descriptors = np.zeros((0, 32), dtype=np.uint8)
        keypoint_coords = np.zeros((0, 2), dtype=np.float32)
    else:
        keypoint_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return FeatureExtractionResult(
        keypoints=list(keypoints) if keypoints else [],
        descriptors=descriptors,
        keypoint_coords=keypoint_coords,
        detector_name="ORB",
    )


def extract_features(
    gray_image: np.ndarray,
    cfg: FeatureConfig,
    use_orb: bool = False,
) -> FeatureExtractionResult:
    """Single entry point used by main.py; chooses detector."""
    if use_orb:
        return extract_orb_features(gray_image, cfg)
    return extract_sift_features(gray_image, cfg)
