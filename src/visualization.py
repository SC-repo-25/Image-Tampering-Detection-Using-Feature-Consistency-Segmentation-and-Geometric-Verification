"""
src/visualization.py
======================
MODULE 5 (part 2) / PHASE 10: Visual Outputs

RESPONSIBILITY
--------------
Save every intermediate and final visual artifact to outputs/visualizations/,
using filenames derived from the input image name so results from multiple
runs don't overwrite each other.

OUTPUTS PRODUCED (per Phase 10 spec):
1. Original image (copied, for convenient side-by-side viewing)
2. Preprocessed image
3. Histogram (matplotlib bar chart, per channel)
4. Feature visualization (keypoints drawn on image)
5. Feature matches (self-match pairs drawn as connecting lines)
6. Candidate suspicious regions (segments containing matches, highlighted)
7. Segmentation mask
8. Geometric verification visualization (inlier matches only)
9. Final tampering heatmap/mask
10. Final result summary (text panel rendered as an image + saved JSON)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend -- no GUI required, per project constraints
import matplotlib.pyplot as plt
import numpy as np

from src.feature_extraction import FeatureExtractionResult
from src.feature_matching import FeatureMatchingResult
from src.geometric_verification import GeometricVerificationResult
from src.histogram_analysis import HistogramAnalysisResult
from src.segmentation import SegmentationResult
from src.tampering_score import TamperingDecision

logger = logging.getLogger(__name__)


def _stem(source_path: Path) -> str:
    return source_path.stem


def save_original_and_preprocessed(
    original_bgr: np.ndarray,
    processed_bgr: np.ndarray,
    source_path: Path,
    vis_dir: Path,
) -> list:
    stem = _stem(source_path)
    paths = []
    p1 = vis_dir / f"{stem}_01_original.png"
    p2 = vis_dir / f"{stem}_02_preprocessed.png"
    cv2.imwrite(str(p1), original_bgr)
    cv2.imwrite(str(p2), processed_bgr)
    return [p1, p2]


def save_histogram_plot(
    histogram_result: HistogramAnalysisResult,
    source_path: Path,
    vis_dir: Path,
) -> Path:
    stem = _stem(source_path)
    out_path = vis_dir / f"{stem}_03_histogram.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["blue", "green", "red"]  # BGR order to match OpenCV convention
    labels = ["B", "G", "R"]
    x = np.arange(histogram_result.global_histogram_bgr.shape[1])
    for c in range(3):
        ax.plot(x, histogram_result.global_histogram_bgr[c], color=colors[c], label=labels[c])
    ax.set_title("Global Color Histogram (normalized)")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_keypoints_visualization(
    processed_bgr: np.ndarray,
    features: FeatureExtractionResult,
    source_path: Path,
    vis_dir: Path,
) -> Path:
    stem = _stem(source_path)
    out_path = vis_dir / f"{stem}_04_keypoints.png"
    vis = cv2.drawKeypoints(
        processed_bgr, features.keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(out_path), vis)
    return out_path


def save_match_visualization(
    processed_bgr: np.ndarray,
    matching_result: FeatureMatchingResult,
    source_path: Path,
    vis_dir: Path,
) -> Path:
    """Draw lines connecting each self-matched keypoint pair, illustrating
    candidate copy-move correspondences before geometric verification.
    """
    stem = _stem(source_path)
    out_path = vis_dir / f"{stem}_05_matches.png"
    vis = processed_bgr.copy()

    for pair in matching_result.match_pairs:
        pt1 = tuple(int(v) for v in pair.query_pt)
        pt2 = tuple(int(v) for v in pair.train_pt)
        cv2.line(vis, pt1, pt2, (0, 165, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 3, (255, 0, 0), -1)

    cv2.imwrite(str(out_path), vis)
    return out_path


def save_segmentation_outputs(
    processed_bgr: np.ndarray,
    segmentation_result: SegmentationResult,
    source_path: Path,
    vis_dir: Path,
) -> list:
    stem = _stem(source_path)
    paths = []

    # Segment boundaries overlay
    seg_map = segmentation_result.segment_map
    boundary_vis = processed_bgr.copy()
    # Compute boundaries via simple gradient of the label map.
    dy = np.diff(seg_map, axis=0, prepend=seg_map[:1])
    dx = np.diff(seg_map, axis=1, prepend=seg_map[:, :1])
    boundaries = (dy != 0) | (dx != 0)
    boundary_vis[boundaries] = (0, 255, 255)
    p1 = vis_dir / f"{stem}_06_segmentation.png"
    cv2.imwrite(str(p1), boundary_vis)
    paths.append(p1)

    # Candidate suspicious region mask
    p2 = vis_dir / f"{stem}_07_candidate_mask.png"
    cv2.imwrite(str(p2), segmentation_result.candidate_mask)
    paths.append(p2)

    return paths


def save_geometric_verification_visualization(
    processed_bgr: np.ndarray,
    geometric_result: GeometricVerificationResult,
    source_path: Path,
    vis_dir: Path,
) -> Path:
    stem = _stem(source_path)
    out_path = vis_dir / f"{stem}_08_geometric_verification.png"
    vis = processed_bgr.copy()

    for pt1, pt2 in zip(geometric_result.inlier_query_pts, geometric_result.inlier_train_pts):
        p1 = tuple(int(v) for v in pt1)
        p2 = tuple(int(v) for v in pt2)
        cv2.line(vis, p1, p2, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # green = RANSAC inlier
        cv2.circle(vis, p1, 4, (0, 255, 0), -1)
        cv2.circle(vis, p2, 4, (0, 255, 0), -1)

    label = "TRANSFORM FOUND" if geometric_result.transform_found else "NO CONSISTENT TRANSFORM"
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), vis)
    return out_path


def save_final_heatmap(
    processed_bgr: np.ndarray,
    segmentation_result: SegmentationResult,
    source_path: Path,
    vis_dir: Path,
) -> Path:
    """Overlay the candidate mask as a translucent red heatmap on the image
    -- the single most important visual for a viva demo.
    """
    stem = _stem(source_path)
    out_path = vis_dir / f"{stem}_09_final_heatmap.png"

    heatmap = np.zeros_like(processed_bgr)
    heatmap[:, :, 2] = segmentation_result.candidate_mask  # red channel

    overlay = cv2.addWeighted(processed_bgr, 0.7, heatmap, 0.6, 0)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def save_summary_panel(
    decision: TamperingDecision,
    source_path: Path,
    vis_dir: Path,
) -> Path:
    stem = _stem(source_path)
    out_path = vis_dir / f"{stem}_10_summary.png"

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    lines = [
        f"Decision: {decision.decision}",
        f"Tampering score: {decision.tampering_score:.4f}",
        f"Match-density component: {decision.match_density_component:.4f}",
        f"Geometric component: {decision.geometric_component:.4f}",
        f"Histogram component: {decision.histogram_component:.4f}",
        f"Matched keypoint pairs: {decision.evidence_summary['num_matched_keypoint_pairs']}",
        f"RANSAC inliers/outliers: "
        f"{decision.evidence_summary['ransac_inliers']}/"
        f"{decision.evidence_summary['ransac_outliers']}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_all_visualizations(
    original_bgr: np.ndarray,
    processed_bgr: np.ndarray,
    histogram_result: HistogramAnalysisResult,
    features: FeatureExtractionResult,
    matching_result: FeatureMatchingResult,
    segmentation_result: SegmentationResult,
    geometric_result: GeometricVerificationResult,
    decision: TamperingDecision,
    source_path: Path,
    vis_dir: Path,
) -> list:
    """Single entry point called by main.py; generates and saves every
    visual artifact, returning the list of saved file paths.
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    saved += save_original_and_preprocessed(original_bgr, processed_bgr, source_path, vis_dir)
    saved.append(save_histogram_plot(histogram_result, source_path, vis_dir))
    saved.append(save_keypoints_visualization(processed_bgr, features, source_path, vis_dir))
    saved.append(save_match_visualization(processed_bgr, matching_result, source_path, vis_dir))
    saved += save_segmentation_outputs(processed_bgr, segmentation_result, source_path, vis_dir)
    saved.append(
        save_geometric_verification_visualization(processed_bgr, geometric_result, source_path, vis_dir)
    )
    saved.append(save_final_heatmap(processed_bgr, segmentation_result, source_path, vis_dir))
    saved.append(save_summary_panel(decision, source_path, vis_dir))

    logger.info("Saved %d visualization files to %s", len(saved), vis_dir)
    return saved
