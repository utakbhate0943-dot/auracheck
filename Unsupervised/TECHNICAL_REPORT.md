# K-Means Unsupervised Modeling: Technical Report

## Overview

This report documents the coding process, dependencies, and techniques used to develop K-Means clustering models for exploring student burnout in the AuraCheck project. The workflow focuses on discovering meaningful student clusters from survey data and comparing those clusters against burnout quartile labels for interpretive context.

---

## 1. Project Objectives

**Primary Goal:** Explore whether student burnout levels can be meaningfully recovered from survey variables using unsupervised learning (K-Means clustering).

**Scope:**
- Perform baseline K-Means clustering on the full feature set
- Compare K-Means variants across multiple dimensions:
  - Number of clusters (k-tuning)
  - Feature subsets (psychosocial, academic/lifestyle)
  - Transformer weights (numeric vs categorical rebalancing)
  - Dimensionality reduction (PCA + K-Means)
  - Stability and reproducibility
- Evaluate cluster quality using internal metrics
- Assess alignment with burnout quartiles for post-hoc interpretation

**Key Assumption:** Burnout quartiles are treated as an interpretive reference only; they do NOT serve as training targets (unsupervised learning).

---

## 2. Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | ≥1.5.0 | K-Means clustering, preprocessing (StandardScaler, OneHotEncoder, ColumnTransformer), dimensionality reduction (PCA), evaluation metrics |
| **pandas** | ≥2.2.0 | Data manipulation, CSV I/O, feature selection |
| **numpy** | ≥1.26.0 | Numerical operations, array handling |
| **matplotlib** | ≥3.9.0 | Static figure generation (elbow plots, heatmaps, scatter plots) |
| **seaborn** | ≥0.13.2 | Statistical data visualization (heatmaps, boxplots) |
| **scipy** | ≥1.13.1 | Scientific computing utilities (as dependency for scikit-learn) |

Full dependency specification: [Unsupervised/requirements-unsupervised.txt](requirements-unsupervised.txt)

---

## 3. Data Preparation

### Data Source
- **File:** `Dataset/students_mental_health_survey_with_burnout_final.csv`
- **Rows:** ~10,000 students
- **Target Variable (for alignment only):** `burnout_raw_score` → converted to quartiles for post-hoc metrics

### Feature Engineering

**Excluded Columns:**
The following columns were excluded to prevent target leakage and redundancy:
- Direct burnout components: `burnout`, `burnout_raw_score`, `burnout_composite_score`
- Stress/mental health targets: `Stress_Level`, `Depression_Score`, `Anxiety_Score`
- Alternative binning schemes: `method1_tertiles`, `method2_wider`, `method3_very_wide`, `method4_manual`, `method5_manual2`, `method6_kmeans`

**Feature Types:**
- **Numeric features:** All numeric columns except burnout-related targets (standardized via `StandardScaler`)
- **Categorical features:** All non-numeric features (one-hot encoded via `OneHotEncoder`)

### Handling Missing Values

**Strategy:**
- **Numeric columns:** Imputed with column median
- **Categorical columns:** Imputed with "Unknown" category

**Timing:** Imputation occurs before preprocessing pipeline to ensure consistent handling across all experiment variants.

---

## 4. Core Modeling Techniques

### 4.1 Data Preprocessing Pipeline

**Technology:** scikit-learn `ColumnTransformer` + `Pipeline`

**Steps:**
1. **Numeric Branch:**
   - Input: All numeric feature columns
   - Operation: `StandardScaler()` (z-score normalization)
   - Output: Scaled features with mean=0, std=1

2. **Categorical Branch:**
   - Input: All categorical feature columns
   - Operation: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
   - Output: One-hot encoded features (dense matrix for scikit-learn compatibility)

3. **Combination:** Features are horizontally concatenated; the resulting matrix is fed into all downstream clustering models.

### 4.2 Baseline K-Means Clustering

**Configuration:**
- **Algorithm:** K-Means (mini-batch with full batch mode)
- **Number of clusters:** 4 (chosen based on k-tuning experiments and interpretability)
- **Initialization:** `n_init=20` (20 random initializations, best solution returned)
- **Random state:** 42 (reproducibility)
- **Max iterations:** 300 (default scikit-learn)
- **Convergence criterion:** Relative change in inertia < 1e-4

**Code Location:** [Unsupervised/scripts/run_kmeans_unsupervised.py](scripts/run_kmeans_unsupervised.py)

### 4.3 K-Means Variants

The experiments script systematically explores K-Means across multiple configurations:
-
#### 4.3.1 K-Tuning (k-sweep)
- **Range:** k ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}
- **Purpose:** Identify the optimal number of clusters based on internal quality metrics
- **Procedure:** Run K-Means for each k value; compute silhouette, davies_bouldin, and calinski_harabasz scores

#### 4.3.2 Feature Subset Experiments
- **Psychosocial Subset:** Features related to mental health, stress, emotions
- **Academic/Lifestyle Subset:** Features related to academic performance, sleep, social engagement
- **Purpose:** Determine whether specific feature domains drive cluster formation differently

#### 4.3.3 Transformer Weight Experiments
- **Default Weights:** Equal contribution from numeric and categorical branches
- **Weighted Variants:** Boost numeric or categorical branches (e.g., 2x numeric weight, 0.5x categorical weight)
- **Purpose:** Assess sensitivity to feature type balance and find optimal feature importance weighting

#### 4.3.4 Dimensionality Reduction + K-Means
- **Technique:** PCA (Principal Component Analysis)
- **PCA Variants:**
  - Retain 95% of variance
  - Retain 90% of variance
  - Fixed n_components (2, 5, 10)
- **Workflow:** Preprocess data → Apply PCA → Feed PCA components to K-Means
- **Purpose:** Reduce noise, improve computational efficiency, discover if clusters are primarily driven by low-rank structure

#### 4.3.5 Stability Analysis
- **Procedure:** Run K-Means k-sweep multiple times with different random seeds (5 seeds)
- **Metric:** Pairwise stability index (average agreement between clusterings)
- **Interpretation:** High stability → robust, reproducible clusters; Low stability → sensitive to initialization

**Code Location:** [Unsupervised/scripts/run_unsupervised_experiments.py](scripts/run_unsupervised_experiments.py)

---

## 5. Evaluation Metrics

All clustering models are evaluated on the preprocessed feature space using both unsupervised (internal) and supervised-style (alignment) metrics.

### 5.1 Unsupervised Quality Metrics

These metrics measure cluster geometry independent of burnout labels.

| Metric | Formula/Reference | Interpretation | Optimal Direction |
|--------|-------------------|-----------------|-------------------|
| **Silhouette Score** | `(b - a) / max(a, b)` per sample | Average measure of cluster cohesion and separation | Higher is better [−1, 1] |
| **Davies-Bouldin Index** | Average max ratio of cluster scatter | Cluster overlap and separation | Lower is better [0, ∞) |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster variance | Compactness and separation | Higher is better [0, ∞) |

### 5.2 Alignment Metrics (Interpretive Reference)

These metrics compare clustering results to burnout quartile labels for qualitative interpretation; they are NOT used during training or model selection.

| Metric | Formula/Reference | Interpretation | Notes |
|--------|-------------------|-----------------|-------|
| **Adjusted Rand Index (ARI)** | Rand index adjusted for chance | Pairwise agreement (−1 to 1) | −1 = opposite, 0 = random, 1 = perfect agreement |
| **Normalized Mutual Information (NMI)** | Shannon entropy-based similarity | Information overlap (0 to 1) | 0 = independent, 1 = identical |

### 5.3 Stability Metric

- **Stability Index:** Average pairwise Adjusted Rand Index across multiple K-Means runs with different seeds
- **Interpretation:** Measures reproducibility and robustness to random initialization

---

## 6. Workflow and Scripts

### 6.1 Execution Workflow

All scripts are designed to be **machine-independent** and resolve paths automatically from the repository root.

**Installation:**
```bash
cd /path/to/auracheck
python -m pip install -r Unsupervised/requirements-unsupervised.txt
python Unsupervised/scripts/verify_unsupervised_setup.py
```

**Execution (from repository root):**

1. **Baseline K-Means:**
   ```bash
   python Unsupervised/scripts/run_kmeans_unsupervised.py
   ```
   Outputs: [Unsupervised/outputs/baseline_kmeans/](outputs/baseline_kmeans/)

2. **K-Means Experiments:**
   ```bash
   python Unsupervised/scripts/run_unsupervised_experiments.py
   ```
   Outputs: [Unsupervised/outputs/kmeans_benchmark/](outputs/kmeans_benchmark/)

3. **Generate PDF Report:**
   ```bash
   python Unsupervised/scripts/export_top_models_pdf.py
   ```
   Outputs: [Unsupervised/outputs/kmeans_benchmark/kmeans_unsupervised_findings_report.pdf](outputs/kmeans_benchmark/kmeans_unsupervised_findings_report.pdf)

### 6.2 Script Overview

#### run_kmeans_unsupervised.py
- **Purpose:** Establish baseline K-Means clustering with k=4
- **Key Steps:**
  1. Load data and build burnout quartile target
  2. Select features (exclude burnout-related columns)
  3. Preprocess: imputation, standardization, one-hot encoding
  4. Fit K-Means with k=4 and n_init=20
  5. Compute metrics: silhouette, davies_bouldin, calinski_harabasz, ARI, NMI
  6. Generate outputs: clustered students CSV, cluster profiles, results JSON
  7. Create visualizations: elbow plot, PCA scatter, heatmaps

#### run_unsupervised_experiments.py
- **Purpose:** Systematically explore K-Means variants across dimensions
- **Key Steps:**
  1. Load data and define feature subsets
  2. For each experiment configuration:
     - Preprocess (with optional transformer weights)
     - Apply PCA if specified
     - Run K-Means k-sweep or stability check
     - Compute metrics
  3. Aggregate results into CSV and JSON
  4. Generate comparison figures (k-sweep metrics, NMI by variant, etc.)

#### export_top_models_pdf.py
- **Purpose:** Synthesize findings into a polished PDF report
- **Key Steps:**
  1. Load baseline and benchmark results
  2. Identify top-performing models by metric
  3. Extract cluster profiles and characteristics
  4. Generate narrative and figures
  5. Export as PDF with embedded images

#### verify_unsupervised_setup.py
- **Purpose:** Validate dependencies and environment
- **Checks:** Import all required packages, verify data file existence, confirm write permissions

---

## 7. Output Artifacts

### 7.1 Baseline K-Means Outputs
**Directory:** [Unsupervised/outputs/baseline_kmeans/](outputs/baseline_kmeans/)

- **kmeans_clustered_students.csv**: Student ID with assigned cluster label
- **kmeans_cluster_profile_summary.csv**: Numeric feature means by cluster
- **kmeans_cluster_profile_categorical.csv**: Categorical feature distributions by cluster
- **kmeans_results.json**: Model parameters and metric values
- **figures/**: Visualizations (elbow plot, PCA scatter, cluster-vs-burnout heatmap, etc.)

### 7.2 Benchmark Experiment Outputs
**Directory:** [Unsupervised/outputs/kmeans_benchmark/](outputs/kmeans_benchmark/)

- **unsupervised_experiments_results.csv**: Results for all k-sweep, feature subset, and variant experiments
- **unsupervised_experiments_results.json**: Detailed configuration and metric values
- **kmeans_unsupervised_findings_report.pdf**: Summary report with findings and key plots
- **figures/**: Comparison plots (k-sweep metrics, top NMI variants, cluster composition, etc.)

---

## 8. Key Design Decisions

### 8.1 Why K-Means Only?
K-Means was chosen for its:
- **Interpretability:** Cluster centers are explicit in feature space
- **Scalability:** Efficient for ~10,000 students
- **Stability:** Well-understood algorithm, reproducible results
- **Simplicity:** Clear hyperparameter (k) for systematic exploration

Alternative clustering methods (e.g., DBSCAN, Gaussian Mixture Models) were deprioritized in favor of a focused K-Means investigation.

### 8.2 Why No Train/Test Split?
K-Means is unsupervised, so there is no train/test paradigm. The model is fit on the entire dataset to maximize learning from all available data. The burnout quartile labels are used only for post-hoc alignment assessment, not for training or hyperparameter tuning.
 r
### 8.3 Why Standardization?
- **Numeric features:** StandardScaler ensures equal feature weight by scale (prevents features with large ranges from dominating the distance metric)
- **Categorical features:** OneHotEncoder converts categorical values to numeric binary columns; no additional scaling needed

### 8.4 Stability via Multiple Initializations
K-Means can converge to local optima. Using `n_init=20` (20 random initializations) and returning the best result reduces this risk and improves reproducibility.

### 8.5 Burnout Quartiles as Interpretive Reference
Burnout is converted to quartiles (rather than continuous or other binning schemes) for:
- **Fairness:** Avoids weighting burnout to either extreme
- **Interpretability:** Provides intuitive groupings ("low," "medium-low," "medium-high," "high")
- **Alignment metrics:** ARI and NMI compare discrete partitions, which is the intended use case

---

## 9. Key Findings and Insights

### 9.1 Cluster Quality
The strongest K-Means-based runs are modest in absolute terms:
- **Silhouette scores** range from ~0.3 to ~0.5 (moderate separation)
- **Davies-Bouldin index** ranges from ~1.5 to ~2.5 (some overlap, but acceptable)
- **Calinski-Harabasz scores** range from ~50 to ~200 (reasonable compactness/separation)

This suggests survey features **only partially** recover meaningful burnout structure in an unsupervised setting.

### 9.2 Burnout Alignment (Weak)
- **Adjusted Rand Index:** Generally < 0.2 (poor alignment to burnout quartiles)
- **Normalized Mutual Information:** Generally < 0.1 (low information overlap)

**Interpretation:** Clusters discovered by K-Means do not strongly align with burnout quartile labels, indicating that burnout is not the primary driver of cluster formation in the feature space.1

### 9.3 Psychosocial Features Perform Well
Feature subset experiments show:
- Psychosocial features (mental health, stress, emotions) produce better geometric clustering than raw feature sets
- Academic/lifestyle subsets often yield poorer internal metrics
- **Implication:** Mental health-related variables are more clusterable than academic performance variables

### 9.4 PCA + K-Means
- Retaining 90–95% variance often improves silhouette and davies_bouldin scores
- Suggests that low-rank structure dominates feature variance
- Can reduce noise and improve interpretability

### 9.5 Stability
- k-sweep stability is generally high (≥0.7) across seeds
- Indicates reproducible cluster assignments
- **Implication:** Results are robust to random initialization

---

## 10. Reproducibility and Future Work

### 10.1 Reproducibility
All scripts use fixed random seeds (`random_state=42`, `n_init=20`) to ensure reproducible results. To re-run:

```bash
python -m pip install -r Unsupervised/requirements-unsupervised.txt
python Unsupervised/scripts/run_kmeans_unsupervised.py
python Unsupervised/scripts/run_unsupervised_experiments.py
python Unsupervised/scripts/export_top_models_pdf.py
```

Expected runtime: ~2–5 minutes (depending on machine hardware).

### 10.2 Potential Enhancements
1. **Supervised dimensionality reduction:** Use LDA or other supervised techniques to boost burnout alignment
2. **Alternative clustering:** Explore hierarchical clustering, DBSCAN, or Gaussian Mixture Models
3. **Feature engineering:** Derive interaction terms or domain-specific aggregates (e.g., "mental health composite")
4. **Hyperparameter search:** Use grid/random search to optimize k, transformer weights, and PCA thresholds
5. **Cross-validation:** Implement k-fold clustering stability checks (requires hold-out validation framework)
6. **Benchmark against baseline:** Compare to simple stratified random clusters or single-feature clustering

---

## 11. References and Further Reading

- **scikit-learn K-Means:** https://scikit-learn.org/stable/modules/clustering.html#k-means
- **scikit-learn Preprocessing:** https://scikit-learn.org/stable/modules/preprocessing.html
- **Silhouette Score:** Rousseeuw, P. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
- **Davies-Bouldin Index:** Davies, D. L., & Bouldin, D. W. (1979). "A cluster separation measure"
- **Calinski-Harabasz Index:** Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster analysis"
- **Adjusted Rand Index:** Hubert, L., & Arabie, P. (1985). "Comparing partitions"
- **Normalized Mutual Information:** Strehl, A., & Ghosh, J. (2002). "Cluster ensembles—A knowledge reuse framework"

---

## 12. Contact and Support

For questions about this workflow or to report issues:
- **Code Location:** [Unsupervised/](.)
- **Documentation:** [Unsupervised/README.md](README.md)
- **Report:** [Unsupervised/outputs/kmeans_benchmark/kmeans_unsupervised_findings_report.pdf](outputs/kmeans_benchmark/kmeans_unsupervised_findings_report.pdf)

---

**Report Generated:** April 20, 2026  
**Last Updated:** April 20, 2026

