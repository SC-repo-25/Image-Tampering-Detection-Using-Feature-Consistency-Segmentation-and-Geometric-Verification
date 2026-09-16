# Viva Preparation

## 2-Minute Project Explanation

"My project detects whether an image has been copy-move tampered — that
is, whether part of the image was duplicated and pasted elsewhere within
the same image — using only classical Computer Vision techniques, no deep
learning black box. The pipeline works in five stages: first I preprocess
the image and check its histogram for color inconsistencies. Then I
extract SIFT keypoints and match the image against itself to find pairs of
regions that look nearly identical but are spatially far apart — candidate
copy-move pairs. Since coincidental similar-looking patches can also match
this way, I run RANSAC to check whether ALL matched pairs agree on one
consistent geometric transform — real copy-move regions do, random
false matches don't. I then use those geometrically-verified matches to
localize the suspicious region with image segmentation, and combine three
pieces of evidence — match density, geometric consistency, and histogram
anomaly — into one transparent, weighted tampering score. Every number in
the final decision can be traced back to an explainable computation."

## 20 Likely Viva Questions & Answers

1. **Why this project?**
   It's a genuine, actively-researched CV problem that naturally exercises
   feature extraction, segmentation, and geometric estimation — the core
   of the syllabus — without needing deep learning or large training data.

2. **Why SIFT and not just Harris corners?**
   Harris finds interesting points but gives no descriptor to compare them
   across the image. SIFT adds a scale/rotation-invariant 128-dim
   descriptor built from local gradient histograms, which is what lets us
   match a region against a possibly resized/rotated copy of itself.

3. **Why is RANSAC needed?**
   Feature matching alone produces false positives — visually similar but
   unrelated patches can match by coincidence. RANSAC checks whether
   matched pairs are explained by ONE consistent geometric transform;
   coincidental matches essentially never agree on a single transform,
   so RANSAC's inlier count is much stronger evidence than raw match count.

4. **What does "feature consistency" mean in this project?**
   Whether local patches across the image have descriptors similar enough,
   at large enough spatial separation, to suggest one was copied from the
   other — measured via the self-matching module's match density.

5. **What does segmentation contribute?**
   It converts a sparse set of matched *points* into a spatially coherent
   *region* — the actual suspicious area — by finding which image segments
   contain the RANSAC-verified matched keypoints.

6. **How is the tampering score calculated?**
   A weighted sum: 0.35 × match_density + 0.45 × geometric_consistency +
   0.20 × histogram_anomaly, each component in [0,1]. Geometric consistency
   is weighted highest because it's the most specific evidence.

7. **What happens when the image has insufficient features?**
   The pipeline degrades gracefully: zero keypoints → zero matches → zero
   score → AUTHENTIC decision, rather than crashing or guessing. Verified
   by `test_image_with_insufficient_features`.

8. **What are the system's limitations?**
   Reliable mainly for copy-move; weak on splicing; texture-dependent
   (fails on flat/low-detail regions); threshold calibrated on a tiny
   6-image dev set; no deep-learning-level robustness to sophisticated
   editing.

9. **Difference between image classification and tampering localization?**
   Classification outputs one label for the whole image with no
   explanation of *where* or *why*. This project outputs a decision AND a
   pixel-level localization mask AND the specific geometric/statistical
   evidence behind it.

10. **Why is this relevant to CSE3010?**
    It applies Module 1 (histograms/filtering), Module 2 (RANSAC/
    homography-style estimation), and Module 3 (SIFT, segmentation)
    together in one coherent, real system rather than as isolated
    exercises.

11. **Why Felzenszwalb segmentation and not K-Means?**
    K-Means clusters purely by color, ignoring spatial/graph structure,
    which tends to produce noisy, disconnected region boundaries.
    Felzenszwalb's graph-based merging (in the same family as Graph Cut)
    respects spatial coherence directly.

12. **Why not use a pretrained deep forensic model?**
    The project's requirement is to demonstrate understanding of classical
    CV mathematics, not to call an opaque model. Every stage here must be
    explainable in this viva.

13. **What is Lowe's ratio test and why use it?**
    It compares the best match's distance to the second-best; a match is
    accepted only if the best is significantly better, rejecting ambiguous
    matches common in repetitive textures.

14. **Why exclude trivial/near self-matches?**
    Every keypoint perfectly matches itself at distance 0; without
    excluding this and nearby points, every image would trivially "match
    itself everywhere."

15. **What's an affine/similarity transform and why use it here?**
    A transform modeling translation, rotation, and uniform scale —
    realistic for how an attacker would paste a region (possibly
    resized/rotated). Estimated here via `cv2.estimateAffinePartial2D`
    with RANSAC.

16. **How did you validate the system actually works?**
    39 automated tests (all passing), plus a real, documented evaluation
    run (83.3% accuracy on a 6-image dev set) with honestly reported
    failure cases (§14 of the report), not fabricated results.

17. **What would improve accuracy most?**
    A larger, real, diverse dataset (CASIA v2/CoMoFoD) for proper
    threshold calibration — the current threshold was tuned on only 6
    images, which is acknowledged as statistically insufficient.

18. **Why weighted sum instead of a trained classifier for fusion?**
    Explainability: a viva examiner can verify the exact arithmetic behind
    any score. A trained classifier would also need a proper labeled
    dataset with train/test separation to avoid overfitting claims.

19. **What's the computational complexity of RANSAC here?**
    Roughly O(k × n) where k is the number of random trials (bounded by
    `ransac_max_iters`) and n is the number of candidate matches — each
    trial fits a transform from 3 points and counts inliers in O(n).

20. **What was the hardest part to get right?**
    Realizing the localization mask was over-flagging the image because it
    used raw pre-RANSAC matches instead of verified inliers — a real
    design bug caught by actually running the pipeline on test data, not
    something visible from code review alone.
