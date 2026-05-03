"""
Train and save the PCA=2 + KMeans(k=4) best unsupervised model.

Artifacts saved:
- pca2_kmeans_preprocessor.joblib (ColumnTransformer: StandardScaler + OneHotEncoder)
- pca2_kmeans_pca.joblib (PCA with n_components=2)
- pca2_kmeans_model.joblib (KMeans with n_clusters=4)
- pca2_kmeans_metadata.json (feature names, classes, config)
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward to find the project root."""
    start_points = [(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for base in start_points:
        for cand in [base, *base.parents]:
            if cand in seen:
                continue
            seen.add(cand)
            if (cand / "Dataset" / "students_mental_health_survey_with_burnout_final.csv").exists():
                return cand
    raise FileNotFoundError("Could not locate repository root")


def main():
    root = find_repo_root()
    data_path = root / "Dataset" / "students_mental_health_survey_with_burnout_final.csv"
    out_dir = root / "Unsupervised" / "outputs" / "pca2_kmeans_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Exclude target/leakage columns
    excluded = {
        "burnout",
        "burnout_raw_score",
        "burnout_composite_score",
        "Stress_Level",
        "Depression_Score",
        "Anxiety_Score",
        "method1_tertiles",
        "method2_wider",
        "method3_very_wide",
        "method4_manual",
        "method5_manual2",
        "method6_kmeans",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Impute missing values
    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna("Unknown")

    # Build preprocessor (StandardScaler + OneHotEncoder)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )

    # Fit preprocessor
    X_proc = preprocessor.fit_transform(X)
    print(f"✓ Preprocessed {X_proc.shape[0]} rows, {X_proc.shape[1]} features")

    # Fit PCA(n_components=2)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_proc)
    print(f"✓ PCA fitted: {X_pca.shape[1]} components, explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Fit KMeans(n_clusters=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_pca)
    print(f"✓ KMeans fitted: {len(np.unique(labels))} clusters found")

    # Save artifacts
    preprocessor_path = out_dir / "pca2_kmeans_preprocessor.joblib"
    pca_path = out_dir / "pca2_kmeans_pca.joblib"
    kmeans_path = out_dir / "pca2_kmeans_model.joblib"
    meta_path = out_dir / "pca2_kmeans_metadata.json"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(pca, pca_path)
    joblib.dump(kmeans, kmeans_path)

    metadata = {
        "model_type": "PCA(2) + KMeans(4)",
        "features_raw": feature_cols,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "n_features_processed": int(X_proc.shape[1]),
        "n_pca_components": 2,
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "n_clusters": 4,
        "cluster_labels": [0, 1, 2, 3],
        "random_state": 42,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved: {preprocessor_path.relative_to(root)}")
    print(f"✓ Saved: {pca_path.relative_to(root)}")
    print(f"✓ Saved: {kmeans_path.relative_to(root)}")
    print(f"✓ Saved: {meta_path.relative_to(root)}")


if __name__ == "__main__":
    main()
