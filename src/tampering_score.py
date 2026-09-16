"""
src/tampering_score.py
========================
MODULE 5 (part 1): Decision & Evaluation -- Score Fusion

RESPONSIBILITY
--------------
- combine evidence from multiple modules
- calculate final tampering score
- classify image as AUTHENTIC or POTENTIALLY TAMPERED
- produce confidence/evidence values

DESIGN PHILOSOPHY
-------------------
This is deliberately a simple, fully transparent WEIGHTED SUM, not a
trained classifier. Reasons:
1. A trained classifier (even simple logistic regression) needs a labeled
   dataset with a defensible train/test split to avoid overfitting claims
   -- out of scope unless Phase 4's dataset work is completed and evaluated
   honestly.
2. A transparent formula means every score is fully explainable in viva:
   "the score is 0.71 because match_density=0.8, geometric_consistency=0.9,
   histogram_anomaly=0.2, weighted as 0.35/0.45/0.20" -- you can justify
   every number.

The three signals fused here map directly to the three analytical pillars
of the project:
  - match_density           <- feature consistency (Module 2)
  - geometric_consistency    <- RANSAC verification (Module 4)
  - histogram_anomaly_score  <- color/illumination consistency (Module 2)

WEIGHT JUSTIFICATION (defaults in config.py):
  geometric_consistency (0.45): weighted highest because it is the
      strongest, most specific evidence -- random matches essentially never
      agree on a single consistent geometric transform.
  match_density (0.35): the number of confidently-matched keypoint pairs;
      necessary but not sufficient (repetitive textures can inflate this
      without real tampering, hence lower weight than the geometry check).
  histogram_anomaly_score (0.20): weakest and noisiest signal (natural
      lighting variation triggers it too), so it is weighted lowest and
      acts mainly as a tie-breaker / supporting signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import ScoringConfig
from src.feature_matching import FeatureMatchingResult
from src.geometric_verification import GeometricVerificationResult
from src.histogram_analysis import HistogramAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class TamperingDecision:
    tampering_score: float       # in [0, 1]
    decision: str                 # "AUTHENTIC" or "POTENTIALLY TAMPERED"
    match_density_component: float
    geometric_component: float
    histogram_component: float
    evidence_summary: dict

    def to_dict(self) -> dict:
        return {
            "tampering_score": round(self.tampering_score, 4),
            "decision": self.decision,
            "components": {
                "match_density": round(self.match_density_component, 4),
                "geometric_consistency": round(self.geometric_component, 4),
                "histogram_anomaly": round(self.histogram_component, 4),
            },
            "evidence_summary": self.evidence_summary,
        }


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_tampering_score(
    matching_result: FeatureMatchingResult,
    geometric_result: GeometricVerificationResult,
    histogram_result: HistogramAnalysisResult,
    cfg: ScoringConfig,
) -> TamperingDecision:
    """Fuse the three evidence signals into one final score and decision.

    MATH: score = w1*match_density + w2*geometric_consistency + w3*histogram_anomaly
    where w1 + w2 + w3 = 1 (weights defined in ScoringConfig). All three
    inputs are already normalized to [0, 1] by their producing modules, so
    the weighted sum is guaranteed to stay in [0, 1] -- no additional
    normalization needed here, but we clip defensively anyway.

    IMPORTANT: match_density can occasionally exceed the useful range on
    very high-texture images; we clip each component to [0, 1] before
    fusing to keep the final score interpretable as a percentage.
    """
    match_density = _clip01(matching_result.match_density)
    geometric_consistency = _clip01(geometric_result.geometric_consistency_score)
    histogram_anomaly = _clip01(histogram_result.histogram_anomaly_score)

    weighted_match = cfg.weight_match_density * match_density
    weighted_geometric = cfg.weight_geometric_consistency * geometric_consistency
    weighted_histogram = cfg.weight_histogram_anomaly * histogram_anomaly

    total_score = _clip01(weighted_match + weighted_geometric + weighted_histogram)

    decision = "POTENTIALLY TAMPERED" if total_score >= cfg.decision_threshold else "AUTHENTIC"

    evidence_summary = {
        "num_matched_keypoint_pairs": matching_result.num_filtered_matches,
        "ransac_transform_found": geometric_result.transform_found,
        "ransac_inliers": geometric_result.num_inliers,
        "ransac_outliers": geometric_result.num_outliers,
        "anomalous_histogram_blocks": len(histogram_result.anomalous_block_indices),
    }

    logger.info(
        "Tampering score=%.4f -> decision=%s "
        "(match=%.3f, geometric=%.3f, histogram=%.3f)",
        total_score, decision, match_density, geometric_consistency, histogram_anomaly,
    )

    return TamperingDecision(
        tampering_score=total_score,
        decision=decision,
        match_density_component=weighted_match,
        geometric_component=weighted_geometric,
        histogram_component=weighted_histogram,
        evidence_summary=evidence_summary,
    )
