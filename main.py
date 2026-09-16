"""
main.py
========
Command-line entry point for the Computer Vision Based Image Tampering
Detection system. No GUI is required or provided -- everything runs from
the terminal, per project constraints.

USAGE
-----
    python main.py --input path/to/image.jpg
    python main.py --input path/to/image.jpg --mode detect --output outputs/
    python main.py --evaluate --dataset data/test/ --output outputs/

The two modes are mutually exclusive:
- Single-image mode (--input): runs the full pipeline on one image and
  prints/saves a detailed report.
- Evaluation mode (--evaluate --dataset): runs the pipeline over a labeled
  dataset folder and prints/saves aggregate classification metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import get_config
from src.evaluation import (
    LABEL_TAMPERED,
    PredictionRecord,
    compute_classification_metrics,
    discover_labeled_images,
)
from src.feature_extraction import extract_features
from src.feature_matching import match_features_self
from src.geometric_verification import verify_geometric_consistency
from src.histogram_analysis import analyze_block_consistency
from src.preprocessing import ImageValidationError, preprocess_image
from src.segmentation import segment_and_localize
from src.tampering_score import compute_tampering_score
from src.visualization import generate_all_visualizations


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline_on_image(image_path: Path, output_dir: Path, cfg, save_visuals: bool = True):
    """Run the complete detection pipeline on a single image.

    Returns (decision, report_dict). Raises ImageValidationError on invalid
    input -- callers are responsible for catching and reporting this
    cleanly.
    """
    print(f"Input image: {image_path}")

    preprocessing_result = preprocess_image(image_path, cfg.preprocessing)
    stats = preprocessing_result.stats
    print(f"Image dimensions: {stats.original_width}x{stats.original_height} "
          f"(processed: {stats.processed_width}x{stats.processed_height})")
    print(f"Preprocessing completed: resized={stats.was_resized}, "
          f"scale_factor={stats.resize_scale_factor:.4f}")

    histogram_result = analyze_block_consistency(preprocessing_result.processed_bgr)
    print(f"Histogram analysis: {len(histogram_result.anomalous_block_indices)} "
          f"anomalous block(s) out of {len(histogram_result.block_stats)}")

    features = extract_features(preprocessing_result.processed_gray, cfg.features)
    print(f"Features detected: {len(features.keypoints)} ({features.detector_name})")

    matching_result = match_features_self(features, cfg.features)
    print(f"Candidate suspicious regions (matched keypoint pairs): "
          f"{matching_result.num_filtered_matches}")

    geometric_result = verify_geometric_consistency(matching_result, cfg.geometric)
    print(f"Geometric verification: transform_found={geometric_result.transform_found}")
    print(f"Inliers: {geometric_result.num_inliers}")
    print(f"Outliers: {geometric_result.num_outliers}")

    # Localization uses RANSAC-INLIER pairs only (geometrically verified),
    # not raw pre-RANSAC matches -- see src/segmentation.py module note for
    # why this ordering matters (raw matches over-flag the image on
    # repetitive-texture content).
    inlier_point_pairs = list(
        zip(geometric_result.inlier_query_pts.tolist(), geometric_result.inlier_train_pts.tolist())
    )
    segmentation_result = segment_and_localize(
        preprocessing_result.processed_bgr, inlier_point_pairs, cfg.segmentation
    )

    decision = compute_tampering_score(
        matching_result, geometric_result, histogram_result, cfg.scoring
    )
    print(f"Tampering score: {decision.tampering_score:.4f}")
    print(f"Final decision: {decision.decision}")

    report = {
        "input_image": str(image_path),
        "preprocessing": stats.to_dict(),
        "histogram_analysis": histogram_result.to_dict(),
        "features": features.to_dict(),
        "feature_matching": matching_result.to_dict(),
        "geometric_verification": geometric_result.to_dict(),
        "segmentation": segmentation_result.to_dict(),
        "decision": decision.to_dict(),
    }

    if save_visuals:
        vis_dir = output_dir / "visualizations"
        saved_paths = generate_all_visualizations(
            preprocessing_result.original_bgr,
            preprocessing_result.processed_bgr,
            histogram_result,
            features,
            matching_result,
            segmentation_result,
            geometric_result,
            decision,
            image_path,
            vis_dir,
        )
        report["visualizations"] = [str(p) for p in saved_paths]

    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{image_path.stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to: {report_path}")

    return decision, report


def run_detect_mode(args, cfg) -> int:
    image_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        run_pipeline_on_image(image_path, output_dir, cfg)
        return 0
    except ImageValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def run_evaluate_mode(args, cfg) -> int:
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1

    labeled_images = discover_labeled_images(dataset_dir)
    print(f"Discovered {len(labeled_images)} labeled image(s) in {dataset_dir}")

    if not labeled_images:
        print(
            "WARNING: No labeled images found. Expected structure:\n"
            f"  {dataset_dir}/authentic/*.jpg\n"
            f"  {dataset_dir}/tampered/*.jpg\n"
            "No metrics can be computed without labeled data.",
            file=sys.stderr,
        )
        return 1

    records = []
    start_time = time.time()

    for idx, (image_path, true_label) in enumerate(labeled_images, start=1):
        print(f"\n[{idx}/{len(labeled_images)}] Evaluating: {image_path.name}")
        try:
            decision, _ = run_pipeline_on_image(image_path, output_dir, cfg, save_visuals=False)
            records.append(
                PredictionRecord(
                    image_path=str(image_path),
                    true_label=true_label,
                    predicted_label=decision.decision,
                    tampering_score=decision.tampering_score,
                )
            )
        except ImageValidationError as e:
            print(f"  SKIPPED (invalid image): {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    metrics = compute_classification_metrics(records)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated: {metrics.num_evaluated}")
    print(f"Elapsed time: {elapsed:.2f}s")
    if metrics.accuracy is not None:
        print(f"Accuracy:  {metrics.accuracy:.4f}")
    if metrics.precision is not None:
        print(f"Precision: {metrics.precision:.4f}")
    if metrics.recall is not None:
        print(f"Recall:    {metrics.recall:.4f}")
    if metrics.f1_score is not None:
        print(f"F1-score:  {metrics.f1_score:.4f}")
    print(
        f"Confusion matrix: TP={metrics.true_positives} FP={metrics.false_positives} "
        f"TN={metrics.true_negatives} FN={metrics.false_negatives}"
    )
    for w in metrics.warnings:
        print(f"NOTE: {w}")

    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_report_path = report_dir / "evaluation_report.json"
    with open(eval_report_path, "w") as f:
        json.dump(
            {
                "metrics": metrics.to_dict(),
                "elapsed_seconds": round(elapsed, 2),
                "records": [
                    {
                        "image_path": r.image_path,
                        "true_label": r.true_label,
                        "predicted_label": r.predicted_label,
                        "tampering_score": round(r.tampering_score, 4),
                    }
                    for r in records
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {eval_report_path}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Computer Vision Based Image Tampering Detection "
        "Using Feature Consistency, Segmentation and Geometric Verification."
    )
    parser.add_argument("--input", type=str, help="Path to a single input image.")
    parser.add_argument(
        "--mode", type=str, default="detect", choices=["analyze", "detect"],
        help="Processing mode for single-image runs (currently both run the same pipeline; "
             "'analyze' is reserved for a future diagnostics-only mode).",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/",
        help="Directory where visualizations and reports are saved.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run in dataset evaluation mode instead of single-image mode.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a labeled dataset directory (required with --evaluate). "
             "Expected layout: <dataset>/authentic/*, <dataset>/tampered/*",
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING",
        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = get_config()

    if args.evaluate:
        if not args.dataset:
            print("ERROR: --evaluate requires --dataset <path>", file=sys.stderr)
            return 1
        return run_evaluate_mode(args, cfg)

    if not args.input:
        print("ERROR: provide --input <image> or use --evaluate --dataset <path>", file=sys.stderr)
        parser.print_help()
        return 1

    return run_detect_mode(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
