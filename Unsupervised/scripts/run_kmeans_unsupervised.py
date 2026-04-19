"""
K-Means unsupervised analysis for student burnout project.

Goal:
- Use all non-target variables to discover student clusters.
- Compare clusters against burnout quartile labels for interpretation.

Outputs:
- Unsupervised/outputs/baseline_kmeans/kmeans_clustered_students.csv
- Unsupervised/outputs/baseline_kmeans/kmeans_cluster_profile_summary.csv
- Unsupervised/outputs/baseline_kmeans/kmeans_cluster_profile_categorical.csv
- Unsupervised/outputs/baseline_kmeans/kmeans_results.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_repo_root(start: Path | None = None) -> Path:
    start_points = [(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for base in start_points:
        for cand in [base, *base.parents]:
            if cand in seen:
                continue
            seen.add(cand)
            if (cand / "Dataset" / "students_mental_health_survey_with_burnout_final.csv").exists():
                return cand
    raise FileNotFoundError(
        "Could not locate repository root containing Dataset/students_mental_health_survey_with_burnout_final.csv"
    )


def build_target(df: pd.DataFrame) -> pd.Series:
    return pd.qcut(
        df["burnout_raw_score"].astype(float),
        q=4,
        labels=[0, 1, 2, 3],
        duplicates="drop",
    ).astype(int)


def main() -> None:
    root = find_repo_root()
    data_path = root / "Dataset" / "students_mental_health_survey_with_burnout_final.csv"
    out_dir = root / "Unsupervised" / "outputs" / "baseline_kmeans"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_clustered = out_dir / "kmeans_clustered_students.csv"
    out_profiles = out_dir / "kmeans_cluster_profile_summary.csv"
    out_profiles_cat = out_dir / "kmeans_cluster_profile_categorical.csv"
    out_results = out_dir / "kmeans_results.json"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    y_true = build_target(df)

    # Keep all available variables except explicit burnout target/derived columns
    # and direct burnout components to avoid leakage.
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

    # Use sparse_output=False for broad compatibility handling fallback.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "encode",
                            ohe,
                        )
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    # simple missing handling before preprocessing pipeline
    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna("Unknown")

    X_proc = preprocess.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_proc)

    # Unsupervised quality metrics
    sil = float(silhouette_score(X_proc, clusters))
    dbi = float(davies_bouldin_score(X_proc, clusters))
    ch = float(calinski_harabasz_score(X_proc, clusters))

    # Alignment with known burnout classes (for interpretation only)
    ari = float(adjusted_rand_score(y_true, clusters))
    nmi = float(normalized_mutual_info_score(y_true, clusters))

    # Map clusters -> burnout class by majority vote for report-friendly pseudo-labels
    map_df = pd.DataFrame({"cluster": clusters, "burnout_q": y_true})
    cluster_to_class = (
        map_df.groupby("cluster")["burnout_q"]
        .agg(lambda s: int(s.value_counts().idxmax()))
        .to_dict()
    )
    predicted_burnout_from_cluster = pd.Series(clusters).map(cluster_to_class).astype(int)

    out_students = df.copy()
    out_students["cluster"] = clusters
    out_students["burnout_q_true"] = y_true
    out_students["burnout_q_from_cluster"] = predicted_burnout_from_cluster
    out_students.to_csv(out_clustered, index=False)

    profile_cols = [c for c in num_cols if c in out_students.columns]
    profile = out_students.groupby("cluster")[profile_cols].mean(numeric_only=True)
    profile["n_students"] = out_students.groupby("cluster").size()
    profile.to_csv(out_profiles)

    # Categorical profile: for each cluster + feature, keep the top category and enrichment (lift).
    cat_profile_rows: list[dict[str, object]] = []
    for c in cat_cols:
        overall_dist = out_students[c].value_counts(normalize=True, dropna=False)
        for cluster_id, sub in out_students.groupby("cluster"):
            cluster_dist = sub[c].value_counts(normalize=True, dropna=False)
            if cluster_dist.empty:
                continue
            top_cat = cluster_dist.index[0]
            p_cluster = float(cluster_dist.iloc[0])
            p_overall = float(overall_dist.get(top_cat, 0.0))
            lift = float(p_cluster / p_overall) if p_overall > 0 else float("nan")
            cat_profile_rows.append(
                {
                    "cluster": int(cluster_id),
                    "feature": c,
                    "top_category": str(top_cat),
                    "cluster_share": p_cluster,
                    "overall_share": p_overall,
                    "lift": lift,
                }
            )

    cat_profile_df = pd.DataFrame(cat_profile_rows)
    if len(cat_profile_df):
        cat_profile_df = cat_profile_df.sort_values(["cluster", "lift"], ascending=[True, False])
    cat_profile_df.to_csv(out_profiles_cat, index=False)

    # --------------------------
    # Visuals for interpretation
    # --------------------------
    sns.set(style="whitegrid")

    # 1) Cluster size distribution
    cluster_counts = out_students["cluster"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=cluster_counts.index.astype(str), y=cluster_counts.values, color="#4C78A8")
    plt.title("K-Means Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    fig_cluster_sizes = fig_dir / "cluster_sizes.png"
    plt.savefig(fig_cluster_sizes, dpi=160)
    plt.close()

    # 2) Cluster vs burnout quartile heatmap
    ct = pd.crosstab(out_students["cluster"], out_students["burnout_q_true"])
    plt.figure(figsize=(7, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Cluster vs Burnout Quartile (True)")
    plt.xlabel("Burnout Quartile")
    plt.ylabel("Cluster")
    plt.tight_layout()
    fig_cluster_vs_burnout = fig_dir / "cluster_vs_burnout_heatmap.png"
    plt.savefig(fig_cluster_vs_burnout, dpi=160)
    plt.close()

    # 3) PCA 2D view colored by cluster
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_proc)
    pca_df = pd.DataFrame({"PC1": pca_2d[:, 0], "PC2": pca_2d[:, 1], "cluster": clusters})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=20, alpha=0.7)
    plt.title("PCA Projection Colored by Cluster")
    plt.tight_layout()
    fig_pca_cluster = fig_dir / "pca_clusters.png"
    plt.savefig(fig_pca_cluster, dpi=160)
    plt.close()

    # 4) PCA 2D view colored by true burnout quartile
    pca_df["burnout_q_true"] = y_true.to_numpy()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="burnout_q_true", palette="viridis", s=20, alpha=0.7)
    plt.title("PCA Projection Colored by True Burnout Quartile")
    plt.tight_layout()
    fig_pca_true = fig_dir / "pca_true_burnout.png"
    plt.savefig(fig_pca_true, dpi=160)
    plt.close()

    # 5) Elbow plot to help choose k
    k_values = list(range(2, 11))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(X_proc)
        inertias.append(float(km.inertia_))

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Plot (Inertia vs Number of Clusters)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    plt.tight_layout()
    fig_elbow = fig_dir / "elbow_plot.png"
    plt.savefig(fig_elbow, dpi=160)
    plt.close()

    metric_explanations = {
        "silhouette": "Measures how well-separated clusters are. Range is -1 to 1; higher is better.",
        "davies_bouldin": "Measures average similarity between clusters. Lower is better.",
        "calinski_harabasz": "Ratio of between-cluster separation to within-cluster compactness. Higher is better.",
        "adjusted_rand_index": "Agreement between cluster assignments and true burnout quartiles (chance-adjusted). Higher is better; 1 is perfect.",
        "normalized_mutual_info": "How much information clusters share with true burnout quartiles. Range is 0 to 1; higher is better.",
    }

    # Human-readable descriptions for each saved figure.
    figure_explanations = {
        "cluster_sizes.png": "Bar chart of number of students per cluster; useful to spot highly imbalanced clusters.",
        "cluster_vs_burnout_heatmap.png": "Heatmap crossing discovered clusters with true burnout quartiles; highlights where clusters align or mix quartiles.",
        "pca_clusters.png": "2D PCA projection colored by cluster label; gives a visual sense of separation/overlap in reduced space.",
        "pca_true_burnout.png": "Same PCA projection colored by true burnout quartiles; compare against cluster-colored view to assess alignment.",
        "elbow_plot.png": "Inertia vs k curve for k=2..10; look for an elbow where adding more clusters yields diminishing returns.",
    }

    payload = {
        "dataset": os.path.relpath(data_path, root),
        "n_rows": int(len(df)),
        "n_features_used": int(len(feature_cols)),
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "kmeans": {
            "n_clusters": 4,
            "random_state": 42,
            "n_init": 20,
        },
        "metrics_unsupervised": {
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
        },
        "alignment_to_burnout_labels": {
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "cluster_to_burnout_class_mapping": {str(k): int(v) for k, v in cluster_to_class.items()},
        },
        "metric_explanations": metric_explanations,
        "figure_explanations": figure_explanations,
        "outputs": {
            "clustered_students_csv": os.path.relpath(out_clustered, root),
            "cluster_profile_summary_csv": os.path.relpath(out_profiles, root),
            "cluster_profile_categorical_csv": os.path.relpath(out_profiles_cat, root),
            "results_json": os.path.relpath(out_results, root),
            "figures": {
                "cluster_sizes": os.path.relpath(fig_cluster_sizes, root),
                "cluster_vs_burnout_heatmap": os.path.relpath(fig_cluster_vs_burnout, root),
                "pca_clusters": os.path.relpath(fig_pca_cluster, root),
                "pca_true_burnout": os.path.relpath(fig_pca_true, root),
                "elbow_plot": os.path.relpath(fig_elbow, root),
            },
        },
    }

    with open(out_results, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", os.path.relpath(out_clustered, root))
    print("Saved:", os.path.relpath(out_profiles, root))
    print("Saved:", os.path.relpath(out_profiles_cat, root))
    print("Saved figures dir:", os.path.relpath(fig_dir, root))
    print("Saved:", os.path.relpath(out_results, root))


if __name__ == "__main__":
    main()
