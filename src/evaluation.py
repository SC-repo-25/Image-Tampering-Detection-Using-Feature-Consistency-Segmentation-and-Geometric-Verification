"""
src/evaluation.py
===================
MODULE 5 (part 3) / PHASE 9: Evaluation Framework

RESPONSIBILITY
--------------
Run the full pipeline over a labeled dataset and compute REAL classification
metrics (accuracy, precision, recall, F1, confusion matrix). No metric here
is ever fabricated -- if the dataset directory is empty or missing, this
module reports that explicitly rather than inventing numbers.

EXPECTED DATASET LAYOUT
-------------------------
data/test/
├── authentic/      <- images known/assumed to be untampered
└── tampered/       <- images known/assumed to be tampered

This mirrors the simplest, most common convention for binary-labeled image
folders (same convention used by torchvision's ImageFolder, scikit-learn
examples, etc.), so it needs no extra label file for the basic case. A CSV
label file is also supported for datasets that don't fit this folder split
(see `load_labels_from_csv`).

METRICS COMPUTED AND WHY
---------------------------
- Accuracy: overall fraction correct. Easy to interpret but misleading if
  classes are imbalanced (e.g. 90 authentic vs 10 tampered) -- hence we also
  report precision/recall/F1.
- Precision (tampered): of images predicted POTENTIALLY TAMPERED, fraction
  that truly are. Relevant if false alarms (flagging authentic images) are
  costly.
- Recall (tampered): of truly tampered images, fraction correctly flagged.
  Relevant if missing real tampering is costly.
- F1: harmonic mean of precision and recall, a single balanced summary.
- Confusion matrix: full breakdown of TP/FP/TN/FN, needed to compute all
  of the above transparently and to spot systematic error patterns.

We do NOT compute IoU/pixel-level localization metrics by default, because
that requires per-image ground-truth tamper MASKS (not just authentic/
tampered labels), which most simple datasets (including a small custom one)
will not have. The framework supports it (`evaluate_localization_iou`) IF
mask files are provided, but never fabricates it otherwise.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LABEL_AUTHENTIC = "AUTHENTIC"
LABEL_TAMPERED = "POTENTIALLY TAMPERED"


@dataclass
class PredictionRecord:
    image_path: str
    true_label: str
    predicted_label: str
    tampering_score: float


@dataclass
class EvaluationResult:
    records: list = field(default_factory=list)   # list[PredictionRecord]
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    num_evaluated: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_evaluated": self.num_evaluated,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
            "warnings": self.warnings,
        }


def discover_labeled_images(dataset_dir: Path) -> list:
    """Discover (image_path, true_label) pairs from the authentic/tampered
    folder convention described in the module docstring.
    """
    pairs = []
    authentic_dir = dataset_dir / "authentic"
    tampered_dir = dataset_dir / "tampered"

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    if authentic_dir.exists():
        for p in sorted(authentic_dir.iterdir()):
            if p.suffix.lower() in valid_ext:
                pairs.append((p, LABEL_AUTHENTIC))

    if tampered_dir.exists():
        for p in sorted(tampered_dir.iterdir()):
            if p.suffix.lower() in valid_ext:
                pairs.append((p, LABEL_TAMPERED))

    return pairs


def load_labels_from_csv(csv_path: Path) -> list:
    """Alternative label source: a CSV with columns `filepath,label`, where
    label is either 'authentic' or 'tampered' (case-insensitive).
    """
    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            true_label = LABEL_TAMPERED if label.startswith("tamp") else LABEL_AUTHENTIC
            pairs.append((Path(row["filepath"]), true_label))
    return pairs


def compute_classification_metrics(records: list) -> EvaluationResult:
    """Compute accuracy/precision/recall/F1/confusion matrix from a list of
    PredictionRecord. "Tampered" is treated as the positive class.
    """
    result = EvaluationResult(records=records, num_evaluated=len(records))

    if not records:
        result.warnings.append(
            "No labeled images were found/evaluated -- metrics cannot be computed."
        )
        return result

    tp = fp = tn = fn = 0
    for r in records:
        pred_positive = r.predicted_label == LABEL_TAMPERED
        true_positive_label = r.true_label == LABEL_TAMPERED

        if pred_positive and true_positive_label:
            tp += 1
        elif pred_positive and not true_positive_label:
            fp += 1
        elif not pred_positive and not true_positive_label:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    if precision is None:
        result.warnings.append(
            "Precision undefined: no images were predicted POTENTIALLY TAMPERED."
        )
    if recall is None:
        result.warnings.append(
            "Recall undefined: no truly tampered images were present in the dataset."
        )

    result.true_positives = tp
    result.false_positives = fp
    result.true_negatives = tn
    result.false_negatives = fn
    result.accuracy = accuracy
    result.precision = precision
    result.recall = recall
    result.f1_score = f1

    return result


def evaluate_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Compute Intersection-over-Union between a predicted binary mask and
    a ground-truth binary mask. ONLY meaningful/called when real
    pixel-level ground truth is available for an image -- never invoked
    with placeholder masks.

    MATH: IoU = |A ∩ B| / |A ∪ B|
    """
    pred_bool = predicted_mask.astype(bool)
    gt_bool = ground_truth_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        # Both masks empty -- perfect agreement on "no tampered region".
        return 1.0

    return float(intersection) / float(union)
