# Unsupervised Clustering Experiment Summary

## Overview

This folder contains results from a KMeans-only comparison of clustering variants for recovering student burnout levels from survey responses.

The current report and ranking tables focus on KMeans-based variants only.

## Key Takeaways

- **Best overall**: KMeans variants with tuned preprocessing/feature spaces can improve geometric separation over the plain baseline.
- **Results are modest**: cluster quality metrics and alignment to burnout quartiles are modest overall, suggesting burnout is only partially recoverable from the available survey features.
- **Dimensionality reduction helps**: PCA-based variants currently improve over plain K-Means, suggesting the data benefits from a compact representation.
- **Stability matters**: repeated runs with different seeds are tracked via `stability_index` for the k-sweep family.
- **Note on data leakage**: the psychosocial feature subset was corrected to exclude Stress_Level, Depression_Score, and Anxiety_Score, which are components of the burnout target. This ensures a fair unsupervised evaluation.

## Method Families Tested

1. **K-Means k-sweep** — tested k=2..10
2. **Feature subsets** — psychosocial and academic/lifestyle subsets
3. **Transformer weights** — rebalanced numeric vs categorical features
4. **PCA + K-Means** — reduced dimensionality before clustering

## How to Read the Results

**Internal metrics** (no labels needed):
- `silhouette`: cluster separation; range -1 to 1; higher is better.
- `davies_bouldin`: cluster overlap; lower is better.
- `calinski_harabasz`: compactness vs separation; higher is better.

**Alignment to burnout** (post-hoc interpretation only):
- `adjusted_rand_index`: agreement with burnout quartiles; 0 is chance, 1 is perfect.
- `normalized_mutual_info`: shared information with burnout quartiles; 0 to 1; higher is better.

**Reproducibility**:
- `stability_index`: average pairwise agreement across repeated K-Means runs with different seeds; higher means more reproducible.

## Output Files

- `unsupervised_experiments_results.csv` — full results table, sorted by NMI.
- `unsupervised_experiments_results.json` — summary with top-10 variants and metadata.
- `figures/` — comparison plots and diagnostic visualizations.

## Files Included

See `Unsupervised/notebooks/kmeans_burnout_unsupervised.ipynb` for a detailed walkthrough of the baseline K-Means workflow and visualization pipeline.
