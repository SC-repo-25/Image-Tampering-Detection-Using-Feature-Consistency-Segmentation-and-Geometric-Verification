# System Diagrams

## 1. High-Level Data Flow

```mermaid
flowchart TD
    A[Input Image] --> B[Validation]
    B --> C[Preprocessing]
    C --> D[Histogram / Block Consistency Analysis]
    C --> E[SIFT Feature Extraction]
    E --> F[Self-Matching + Lowe's Ratio Test]
    F --> G[RANSAC Geometric Verification]
    G --> H[Segmentation restricted to RANSAC inliers]
    D --> I[Tampering Score Fusion]
    F --> I
    G --> I
    I --> J{Score >= threshold?}
    J -->|Yes| K[POTENTIALLY TAMPERED]
    J -->|No| L[AUTHENTIC]
    H --> M[Visual + JSON Report]
    I --> M
```

## 2. Module / Component Diagram

```mermaid
classDiagram
    class Preprocessing {
        +preprocess_image(path, cfg)
        +load_image()
        +resize_if_needed()
        +reduce_noise()
    }
    class HistogramAnalysis {
        +analyze_block_consistency()
        +compute_global_histogram()
        +chi_square_distance()
    }
    class FeatureExtraction {
        +extract_sift_features()
        +extract_orb_features()
    }
    class FeatureMatching {
        +match_features_self()
    }
    class GeometricVerification {
        +verify_geometric_consistency()
    }
    class Segmentation {
        +segment_and_localize()
        +identify_candidate_segments()
        +build_candidate_mask()
    }
    class TamperingScore {
        +compute_tampering_score()
    }
    class Visualization {
        +generate_all_visualizations()
    }
    class Evaluation {
        +compute_classification_metrics()
        +evaluate_iou()
    }
    class MainCLI {
        +run_pipeline_on_image()
        +run_detect_mode()
        +run_evaluate_mode()
    }

    MainCLI --> Preprocessing
    MainCLI --> HistogramAnalysis
    MainCLI --> FeatureExtraction
    MainCLI --> FeatureMatching
    MainCLI --> GeometricVerification
    MainCLI --> Segmentation
    MainCLI --> TamperingScore
    MainCLI --> Visualization
    MainCLI --> Evaluation
    FeatureMatching --> FeatureExtraction
    GeometricVerification --> FeatureMatching
    Segmentation --> GeometricVerification
    TamperingScore --> FeatureMatching
    TamperingScore --> GeometricVerification
    TamperingScore --> HistogramAnalysis
```

## 3. Sequence Diagram (single-image `detect` run)

```mermaid
sequenceDiagram
    participant User
    participant CLI as main.py
    participant Pre as preprocessing.py
    participant Hist as histogram_analysis.py
    participant Feat as feature_extraction.py
    participant Match as feature_matching.py
    participant Geo as geometric_verification.py
    participant Seg as segmentation.py
    participant Score as tampering_score.py
    participant Vis as visualization.py

    User->>CLI: python main.py --input img.jpg
    CLI->>Pre: preprocess_image(path)
    Pre-->>CLI: PreprocessingResult
    CLI->>Hist: analyze_block_consistency(image)
    Hist-->>CLI: HistogramAnalysisResult
    CLI->>Feat: extract_features(gray)
    Feat-->>CLI: FeatureExtractionResult
    CLI->>Match: match_features_self(features)
    Match-->>CLI: FeatureMatchingResult
    CLI->>Geo: verify_geometric_consistency(matches)
    Geo-->>CLI: GeometricVerificationResult
    CLI->>Seg: segment_and_localize(image, inlier_pairs)
    Seg-->>CLI: SegmentationResult
    CLI->>Score: compute_tampering_score(...)
    Score-->>CLI: TamperingDecision
    CLI->>Vis: generate_all_visualizations(...)
    Vis-->>CLI: saved file paths
    CLI-->>User: console report + JSON + visualizations
```

## 4. Use Case Diagram

```mermaid
flowchart LR
    Student((Student / Evaluator))
    Student --> UC1[Analyze single image]
    Student --> UC2[Batch-evaluate labeled dataset]
    Student --> UC3[Inspect visualizations]
    Student --> UC4[Review JSON report]
    UC1 --> Sys[Tampering Detection System]
    UC2 --> Sys
    UC3 --> Sys
    UC4 --> Sys
```

## 5. Algorithm Pipeline (evidence fusion detail)

```mermaid
flowchart LR
    subgraph Evidence Collection
        M[Match density] 
        G[Geometric consistency]
        H[Histogram anomaly]
    end
    M -->|weight 0.35| S[Weighted Sum]
    G -->|weight 0.45| S
    H -->|weight 0.20| S
    S --> T{score >= 0.10?}
    T -->|yes| TP[POTENTIALLY TAMPERED]
    T -->|no| AU[AUTHENTIC]
```
