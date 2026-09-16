# statement.md

## Project Title
Computer Vision Based Image Tampering Detection Using Feature Consistency, Segmentation and Geometric Verification

## Problem Statement
Given a single digital image with no reference original and no trusted
metadata, determine whether the image is likely authentic or has been
manipulated (specifically, copy-move forgery), and localize the suspicious
region(s), using only classical Computer Vision techniques — no end-to-end
deep learning black box.

## Scope
- Single-image, blind (no reference image) tampering analysis.
- Primary target: **copy-move forgery** (a region duplicated within the
  same image), which is reliably detectable with self-similarity feature
  matching and geometric verification.
- Secondary, best-effort signal: global/local color and histogram
  inconsistency, useful as weak supporting evidence for splicing but not
  claimed as a strong standalone detector.
- Command-line only; no GUI.
- Classical CV pipeline: every stage inspectable and explainable, no
  pretrained deep forensic model used for the core decision.

## Target Users
Computer Vision students, course evaluators, and anyone wanting an
interpretable, from-first-principles forensic analysis tool for
educational or demonstrative purposes — not a production-grade
forensic/legal tool.

## High-Level Features
- Image validation and preprocessing (resize, denoise, grayscale)
- Block-wise histogram/color consistency analysis
- SIFT keypoint extraction
- Self-matching for copy-move candidate pair detection (Lowe's ratio test)
- Felzenszwalb graph-based segmentation for candidate region localization
- RANSAC-based affine transform verification of matched region pairs
- Transparent weighted-sum tampering score fusion
- Full visual report (10 saved images per analyzed image)
- Machine-readable JSON report per image
- Batch evaluation mode with real accuracy/precision/recall/F1 metrics

## Expected Inputs
A single RGB image file (.jpg, .jpeg, .png, .bmp, .tif, .tiff), or a
labeled dataset directory (`<dataset>/authentic/`, `<dataset>/tampered/`)
for evaluation mode.

## Expected Outputs
- Console report of every pipeline stage's real, computed values
- `outputs/visualizations/<image>_NN_<stage>.png` — 10 visual artifacts
- `outputs/reports/<image>_report.json` — full machine-readable results
- (evaluation mode) `outputs/reports/evaluation_report.json` — aggregate
  accuracy/precision/recall/F1/confusion matrix

## Main Technologies
Python 3, OpenCV, NumPy, SciPy, scikit-image, Matplotlib, pytest.

## Major Computer Vision Concepts Used
Gaussian filtering, histogram analysis, SIFT (scale-space, DoG, gradient
orientation histograms), feature matching with Lowe's ratio test,
graph-based image segmentation, affine transform estimation, RANSAC robust
estimation, morphological image processing.
