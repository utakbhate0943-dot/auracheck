# Unsupervised

This folder contains unsupervised modeling work for student burnout.

## Goal

Explore whether student burnout levels can be meaningfully recovered from the available survey variables using unsupervised learning.

The current workflow treats burnout quartiles as an interpretive reference only. Cluster quality is measured with internal metrics, and alignment to burnout quartiles is used as a post-hoc comparison.

For the current report and comparison tables, the focus is on KMeans-based variants only.

## Experiment families

- **K-Means k-sweep** across `k=2..10`
- **Feature subset experiments** using psychosocial and academic/lifestyle subsets
- **Transformer-weight experiments** that rebalance numeric vs categorical features
- **PCA + K-Means** variants
- **Stability checks** on k-sweep runs to evaluate reproducibility across seeds

## Structure
- `notebooks/kmeans_burnout_unsupervised.ipynb` — notebook workflow
- `scripts/run_kmeans_unsupervised.py` — baseline K-Means script
- `scripts/run_unsupervised_experiments.py` — expanded experiment runner
- `outputs/kmeans/` — baseline K-Means outputs
- `outputs/experiments/` — comparative experiment outputs

## Outputs
- Baseline K-Means artifacts:
	- `kmeans_clustered_students.csv`
	- `kmeans_cluster_profile_summary.csv`
	- `kmeans_cluster_profile_categorical.csv`
	- `kmeans_cluster_profile_categorical_strong_signals.csv`
	- `kmeans_results.json`
- Comparative experiment artifacts:
	- `unsupervised_experiments_results.csv`
	- `unsupervised_experiments_results.json`
	- `unsupervised_metric_explanations.txt`
	- `unsupervised_figure_explanations.txt`
	- `kmeans_unsupervised_findings_report.pdf`
	- `figures/*.png`

## How to read the results

- `silhouette`: cluster separation; higher is better.
- `davies_bouldin`: cluster overlap; lower is better.
- `calinski_harabasz`: compactness vs separation; higher is better.
- `adjusted_rand_index` and `normalized_mutual_info`: alignment to burnout quartiles for interpretation, not training.
- `stability_index`: average pairwise agreement across repeated K-Means runs; higher means more reproducible clusters.

## Machine-independent run notes

All scripts resolve paths from the repository root and do not require machine-specific paths.

Install folder-specific dependencies (from repository root):

- `python -m pip install -r Unsupervised/requirements-unsupervised.txt`
- `python Unsupervised/scripts/verify_unsupervised_setup.py`

From repository root:

- `python Unsupervised/scripts/run_kmeans_unsupervised.py`
- `python Unsupervised/scripts/run_unsupervised_experiments.py`
- `python Unsupervised/scripts/export_top_models_pdf.py`

## Current takeaway

The strongest KMeans-based runs are still modest in absolute terms, which suggests the survey features only partially recover burnout quartiles. Psychosocial and PCA variants often improve geometric separation, but burnout-label alignment (ARI/NMI) remains weak.
