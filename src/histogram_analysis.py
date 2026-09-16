"""
src/histogram_analysis.py
==========================
PART OF MODULE 2: Image Feature & Consistency Analysis

RESPONSIBILITY
--------------
Analyze the image's histogram and block-wise color/illumination statistics
to detect regions whose color distribution is inconsistent with the rest of
the image. This is a classic, well-documented cue in splicing forensics:
content pasted from a different source image often has subtly different
white balance, exposure, or sensor color response than its surroundings,
even after careful retouching.

WHERE THIS SITS IN THE PIPELINE
--------------------------------
Preprocessing -> [THIS MODULE] -> Feature Extraction -> ...

SYLLABUS MAPPING
-----------------
Module 1: Histogram processing, image enhancement/restoration concepts.

WHAT THIS MODULE DOES NOT CLAIM
---------------------------------
Histogram anomalies alone are a WEAK signal -- lots of authentic images have
locally varying lighting (shadows, windows, mixed light sources). This
module produces a numerical "histogram anomaly score" that is only ONE of
three inputs later fused by tampering_score.py. It is never used alone to
make a decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Number of histogram bins per channel. 32 is a standard, well-tested choice:
# coarse enough to compare noisy real-world distributions robustly, fine
# enough to catch a genuine shift in color balance.
HIST_BINS = 32
HIST_RANGE = (0, 256)


@dataclass
class BlockStats:
    """Color statistics for one spatial block of the image."""

    row: int
    col: int
    mean_bgr: tuple
    std_bgr: tuple
    chi_square_distance: float  # distance of this block's histogram from the global histogram


@dataclass
class HistogramAnalysisResult:
    global_histogram_bgr: np.ndarray       # shape (3, HIST_BINS)
    block_grid_shape: tuple                # (rows, cols) of the block grid
    block_stats: list                      # list[BlockStats]
    anomalous_block_indices: list           # list[(row, col)] flagged as outliers
    histogram_anomaly_score: float          # in [0, 1], fused into final tampering score

    def to_dict(self) -> dict:
        return {
            "block_grid_shape": self.block_grid_shape,
            "num_blocks": len(self.block_stats),
            "num_anomalous_blocks": len(self.anomalous_block_indices),
            "anomalous_block_indices": self.anomalous_block_indices,
            "histogram_anomaly_score": round(self.histogram_anomaly_score, 4),
        }


def compute_global_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized per-channel color histogram for the whole image.

    MATH: For each channel c, hist_c[b] = (# pixels in channel c whose value
    falls in bin b) / (total pixels). Normalizing by pixel count makes
    histograms comparable between the whole image and much smaller blocks,
    since raw counts would otherwise scale with block area.
    """
    hist = np.zeros((3, HIST_BINS), dtype=np.float64)
    for c in range(3):
        h = cv2.calcHist([image_bgr], [c], None, [HIST_BINS], HIST_RANGE)
        h = h.flatten()
        total = h.sum()
        hist[c] = h / total if total > 0 else h
    return hist


def chi_square_distance(hist_a: np.ndarray, hist_b: np.ndarray, eps: float = 1e-10) -> float:
    """Chi-square distance between two histograms, averaged over channels.

    MATH: chi2(A, B) = 0.5 * sum_i [ (A_i - B_i)^2 / (A_i + B_i + eps) ]
    This is a standard histogram-comparison distance (more sensitive to
    differences in low-count bins than Euclidean distance would be), widely
    used in texture/color retrieval and forensic block comparison. Result is
    0 for identical histograms and grows as distributions diverge; it has no
    fixed upper bound, so it is later normalized relative to the block
    population, not to an assumed absolute maximum.
    """
    distances = []
    for c in range(hist_a.shape[0]):
        a, b = hist_a[c], hist_b[c]
        d = 0.5 * np.sum(((a - b) ** 2) / (a + b + eps))
        distances.append(d)
    return float(np.mean(distances))


def analyze_block_consistency(
    image_bgr: np.ndarray,
    block_size: int = 64,
    outlier_z_threshold: float = 2.5,
) -> HistogramAnalysisResult:
    """Divide the image into a grid of blocks, compute each block's color
    histogram, and flag blocks whose histogram is a statistical outlier
    relative to the population of all blocks.

    APPROACH:
    1. Compute the global histogram (used for reporting, not for outlier
       detection directly -- comparing every block only to the global
       average would flag large authentic regions like a sky or wall as
       "anomalous" purely because they dominate the image).
    2. Compute each block's chi-square distance to the OTHER blocks' mean
       histogram (leave-one-out avoids the block biasing its own baseline).
    3. Flag blocks whose chi-square distance is a statistical outlier
       (z-score beyond `outlier_z_threshold`) relative to the distribution
       of all block distances. This adapts to each image instead of using a
       fixed absolute threshold, which would fail differently on a
       low-contrast vs. a high-contrast photo.

    Args:
        block_size: side length in pixels of each square block.
        outlier_z_threshold: number of standard deviations above the mean
            distance required for a block to be flagged. 2.5 is a
            conventional statistical outlier cutoff (roughly 99th
            percentile for a normal distribution).
    """
    height, width = image_bgr.shape[:2]
    rows = max(1, height // block_size)
    cols = max(1, width // block_size)

    block_histograms = []
    block_positions = []
    block_means = []
    block_stds = []

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * block_size, min((r + 1) * block_size, height)
            x0, x1 = c * block_size, min((c + 1) * block_size, width)
            block = image_bgr[y0:y1, x0:x1]

            if block.size == 0:
                continue

            block_hist = compute_global_histogram(block)
            block_histograms.append(block_hist)
            block_positions.append((r, c))
            block_means.append(tuple(float(x) for x in block.reshape(-1, 3).mean(axis=0)))
            block_stds.append(tuple(float(x) for x in block.reshape(-1, 3).std(axis=0)))

    n_blocks = len(block_histograms)
    if n_blocks < 2:
        logger.warning("Image too small for block-wise histogram analysis; skipping.")
        global_hist = compute_global_histogram(image_bgr)
        return HistogramAnalysisResult(
            global_histogram_bgr=global_hist,
            block_grid_shape=(rows, cols),
            block_stats=[],
            anomalous_block_indices=[],
            histogram_anomaly_score=0.0,
        )

    # Leave-one-out mean histogram per block, and this block's distance to it.
    stacked = np.stack(block_histograms, axis=0)  # (n_blocks, 3, HIST_BINS)
    total_sum = stacked.sum(axis=0)

    distances = []
    for i in range(n_blocks):
        others_mean = (total_sum - stacked[i]) / (n_blocks - 1)
        d = chi_square_distance(stacked[i], others_mean)
        distances.append(d)

    distances = np.array(distances)
    mean_d, std_d = distances.mean(), distances.std()
    z_scores = (distances - mean_d) / (std_d + 1e-10)

    anomalous_indices = []
    block_stats_list = []
    for i in range(n_blocks):
        r, c = block_positions[i]
        block_stats_list.append(
            BlockStats(
                row=r, col=c,
                mean_bgr=block_means[i],
                std_bgr=block_stds[i],
                chi_square_distance=float(distances[i]),
            )
        )
        if z_scores[i] > outlier_z_threshold:
            anomalous_indices.append((r, c))

    # Anomaly score: fraction of blocks flagged as outliers, in [0, 1].
    # A simple, interpretable, and honest metric -- explicitly NOT a
    # probability of tampering by itself.
    anomaly_score = len(anomalous_indices) / n_blocks

    global_hist = compute_global_histogram(image_bgr)

    return HistogramAnalysisResult(
        global_histogram_bgr=global_hist,
        block_grid_shape=(rows, cols),
        block_stats=block_stats_list,
        anomalous_block_indices=anomalous_indices,
        histogram_anomaly_score=float(anomaly_score),
    )
