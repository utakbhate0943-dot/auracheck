"""
Inference script: load PCA=2 + KMeans model and make predictions on new survey data.

Usage:
    python predict_pca2_kmeans.py --input new_surveys.csv --output predictions.csv

Or use directly in Python:
    from predict_pca2_kmeans import predict_batch
    preds = predict_batch("path/to/new_surveys.csv")
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward to find the project root."""
    start_points = [(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for base in start_points:
        for cand in [base, *base.parents]:
            if cand in seen:
                continue
            seen.add(cand)
            if (cand / "Unsupervised" / "outputs" / "pca2_kmeans_model").exists():
                return cand
    raise FileNotFoundError("Could not locate repository root with pca2_kmeans_model")


def load_model_artifacts(model_dir: Path) -> tuple[Any, Any, Any, dict]:
    """Load preprocessor, PCA, KMeans, and metadata."""
    preprocessor = joblib.load(model_dir / "pca2_kmeans_preprocessor.joblib")
    pca = joblib.load(model_dir / "pca2_kmeans_pca.joblib")
    kmeans = joblib.load(model_dir / "pca2_kmeans_model.joblib")

    with open(model_dir / "pca2_kmeans_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return preprocessor, pca, kmeans, metadata


def preprocess_new_data(df: pd.DataFrame, preprocessor: Any, metadata: dict) -> np.ndarray:
    """Apply the same preprocessing as training: scale numerics, one-hot categoricals."""
    feature_cols = metadata["features_raw"]
    X = df[feature_cols].copy()

    num_cols = metadata["numeric_features"]
    cat_cols = metadata["categorical_features"]

    # Impute missing values
    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna("Unknown")

    # Transform using fitted preprocessor
    X_proc = preprocessor.transform(X)
    return X_proc


def compute_cluster_metrics(df_new: pd.DataFrame, x_pca: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    """Compute unsupervised metrics and optional alignment metrics when labels exist."""
    metrics: dict[str, float] = {}

    unique_clusters = np.unique(clusters)
    if len(unique_clusters) > 1 and len(df_new) > len(unique_clusters):
        metrics["silhouette"] = float(silhouette_score(x_pca, clusters))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x_pca, clusters))
        metrics["davies_bouldin"] = float(davies_bouldin_score(x_pca, clusters))

    # Optional: if target label exists in the input, compute alignment metrics.
    if "burnout_raw_score" in df_new.columns:
        y_true = pd.qcut(
            df_new["burnout_raw_score"].astype(float),
            q=4,
            labels=[0, 1, 2, 3],
            duplicates="drop",
        ).astype(int)
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(y_true, clusters))
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(y_true, clusters))

    return metrics


def predict_batch_with_metrics(input_path: str, model_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Like predict_batch, but also returns clustering metrics for the input batch."""
    if model_dir is None:
        root = find_repo_root()
        model_dir = root / "Unsupervised" / "outputs" / "pca2_kmeans_model"

    preprocessor, pca, kmeans, metadata = load_model_artifacts(model_dir)

    df_new = pd.read_csv(input_path)
    X_proc = preprocess_new_data(df_new, preprocessor, metadata)
    X_pca = pca.transform(X_proc)
    clusters = kmeans.predict(X_pca)

    result = df_new.copy()
    result["cluster_pca2_kmeans"] = clusters

    metrics = compute_cluster_metrics(df_new, X_pca, clusters)
    return result, metrics


def predict_batch(input_path: str, model_dir: Path | None = None) -> pd.DataFrame:
    """
    Load new survey data and make cluster predictions.

    Args:
        input_path: Path to CSV with new survey responses
        model_dir: Optional path to model directory (auto-detected if not provided)

    Returns:
        DataFrame with original data + cluster assignment
    """
    result, _ = predict_batch_with_metrics(input_path, model_dir)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict clusters for new survey data using PCA=2 + KMeans model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with new survey responses",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV (default: input_pca2_kmeans_predictions.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to save clustering metrics as JSON",
    )

    args = parser.parse_args()

    input_path = args.input
    model_dir = Path(args.model_dir) if args.model_dir else None
    output_path = args.output or input_path.replace(".csv", "_pca2_kmeans_predictions.csv")

    print(f"Loading input from: {input_path}")
    result_df, metrics = predict_batch_with_metrics(input_path, model_dir)

    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")
    print(f"  Total rows: {len(result_df)}")
    print(f"  Cluster distribution:\n{result_df['cluster_pca2_kmeans'].value_counts().sort_index()}")

    if metrics:
        print("\n  Metrics:")
        for key, value in metrics.items():
            print(f"    - {key}: {value:.6f}")

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
