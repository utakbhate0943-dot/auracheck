"""
Run a KMeans-focused unsupervised experiment suite for student burnout clustering.

Covers:
1) KMeans k-tuning (k=2..10)
2) Feature subset and transformer-weight variants
3) Dimensionality reduction variants (PCA) + KMeans
4) Stability checks for KMeans k-sweep variants

Outputs under Unsupervised/outputs/kmeans_benchmark:
- unsupervised_experiments_results.csv
- unsupervised_experiments_results.json
- figures/*.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from the current folder to find the project root."""
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
    """Build a pseudo-label target by splitting burnout score into quartiles.

    This is used only for evaluation/alignment metrics (ARI/NMI), not as model input.
    """
    return pd.qcut(
        df["burnout_raw_score"].astype(float),
        q=4,
        labels=[0, 1, 2, 3],
        duplicates="drop",
    ).astype(int)


def preprocess_frame(
    X_raw: pd.DataFrame,
    transformer_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Prepare mixed-type features for KMeans.

    Steps:
    1) impute numeric/categorical missing values,
    2) scale numeric columns,
    3) one-hot encode categoricals,
    4) optionally apply branch-specific transformer weights.
    """
    X = X_raw.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna("Unknown")

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        transformer_weights=transformer_weights,
    )

    X_proc = preprocess.fit_transform(X)
    return X_proc, num_cols, cat_cols


def metric_bundle(X_proc: np.ndarray, labels: np.ndarray, y_true: pd.Series) -> dict[str, float]:
    """Compute internal clustering quality + alignment metrics."""
    out: dict[str, float] = {}
    unique = np.unique(labels)

    if len(unique) >= 2:
        out["silhouette"] = float(silhouette_score(X_proc, labels))
        out["davies_bouldin"] = float(davies_bouldin_score(X_proc, labels))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_proc, labels))
    else:
        out["silhouette"] = float("nan")
        out["davies_bouldin"] = float("nan")
        out["calinski_harabasz"] = float("nan")

    out["adjusted_rand_index"] = float(adjusted_rand_score(y_true, labels))
    out["normalized_mutual_info"] = float(normalized_mutual_info_score(y_true, labels))
    return out


def add_result(results: list[dict[str, Any]], base: dict[str, Any], metrics: dict[str, float]) -> None:
    """Append one model run (metadata + metrics) to results."""
    row = dict(base)
    row.update(metrics)
    results.append(row)


def json_safe(value: Any) -> Any:
    """Recursively convert NaN/Inf values into JSON-safe nulls."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def run_stability_index(
    X_proc: np.ndarray,
    n_clusters: int,
    seeds: list[int],
    n_init: int = 20,
) -> float:
    """Average pairwise ARI across multiple KMeans runs (higher = more stable)."""
    labels_runs: list[np.ndarray] = []
    for seed in seeds:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
        labels_runs.append(km.fit_predict(X_proc))

    pair_aris: list[float] = []
    for i in range(len(labels_runs)):
        for j in range(i + 1, len(labels_runs)):
            pair_aris.append(float(adjusted_rand_score(labels_runs[i], labels_runs[j])))

    if not pair_aris:
        return float("nan")
    return float(np.mean(pair_aris))


def main() -> None:
    root = find_repo_root()
    data_path = root / "Dataset" / "students_mental_health_survey_with_burnout_final.csv"

    out_dir = root / "Unsupervised" / "outputs" / "kmeans_benchmark"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "unsupervised_experiments_results.csv"
    out_json = out_dir / "unsupervised_experiments_results.json"

    df = pd.read_csv(data_path)
    y_true = build_target(df)

    # Exclude direct target/leakage columns from clustering inputs.
    # Stress/Depression/Anxiety are intentionally excluded to keep clustering label-agnostic.
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
    feature_cols_all = [c for c in df.columns if c not in excluded]
    X_all = df[feature_cols_all].copy()

    # Global preprocessing baseline used by most experiment families.
    X_full_proc, _, _ = preprocess_frame(X_all)

    results: list[dict[str, Any]] = []

    stability_seeds = [11, 17, 23, 31, 47]

    # 1) KMeans k sweep
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_full_proc)
        stability_idx = run_stability_index(X_full_proc, n_clusters=k, seeds=stability_seeds)
        add_result(
            results,
            {
                "family": "kmeans_k_sweep",
                "variant": f"k={k}",
                "k": k,
                "n_rows": int(len(df)),
                "n_features_raw": int(len(feature_cols_all)),
                "n_features_processed": int(X_full_proc.shape[1]),
                "stability_index": stability_idx,
            },
            metric_bundle(X_full_proc, labels, y_true),
        )

    # 2) Feature subset variants
    # Compare full feature space vs domain-specific subsets.
    feature_sets = {
        "all_features": feature_cols_all,
        "psychosocial_focus": [
            c
            for c in [
                "Sleep_Quality",
                "Physical_Activity",
                "Diet_Quality",
                "Social_Support",
                "Substance_Use",
                "Counseling_Service_Use",
                "Family_History",
                "Chronic_Illness",
                "Financial_Stress",
            ]
            if c in feature_cols_all
        ],
        "academic_lifestyle_focus": [
            c
            for c in [
                "Course",
                "CGPA",
                "Semester_Credit_Load",
                "Extracurricular_Involvement",
                "Residence_Type",
                "Relationship_Status",
                "Age",
                "Gender",
            ]
            if c in feature_cols_all
        ],
    }

    for name, cols in feature_sets.items():
        if len(cols) < 2:
            continue
        X_sub_proc, _, _ = preprocess_frame(df[cols])
        km = KMeans(n_clusters=4, random_state=42, n_init=20)
        labels = km.fit_predict(X_sub_proc)
        add_result(
            results,
            {
                "family": "kmeans_feature_subset",
                "variant": name,
                "k": 4,
                "n_rows": int(len(df)),
                "n_features_raw": int(len(cols)),
                "n_features_processed": int(X_sub_proc.shape[1]),
            },
            metric_bundle(X_sub_proc, labels, y_true),
        )

    # 2b) Transformer-weight variants
    # Change relative influence of numeric vs categorical blocks after preprocessing.
    for name, tw in {
        "num_heavy": {"num": 1.5, "cat": 1.0},
        "cat_heavy": {"num": 1.0, "cat": 1.5},
    }.items():
        X_w_proc, _, _ = preprocess_frame(X_all, transformer_weights=tw)
        km = KMeans(n_clusters=4, random_state=42, n_init=20)
        labels = km.fit_predict(X_w_proc)
        add_result(
            results,
            {
                "family": "kmeans_transformer_weight",
                "variant": name,
                "k": 4,
                "n_rows": int(len(df)),
                "n_features_raw": int(len(feature_cols_all)),
                "n_features_processed": int(X_w_proc.shape[1]),
            },
            metric_bundle(X_w_proc, labels, y_true),
        )

    # 3) PCA variants + KMeans
    # Evaluate whether lower-dimensional latent structure improves separation/alignment.
    for n_comp in [2, 5, 10, 20]:
        if n_comp > X_full_proc.shape[1]:
            continue
        X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X_full_proc)
        km = KMeans(n_clusters=4, random_state=42, n_init=20)
        labels = km.fit_predict(X_pca)
        add_result(
            results,
            {
                "family": "kmeans_pca",
                "variant": f"pca={n_comp}",
                "k": 4,
                "n_rows": int(len(df)),
                "n_features_raw": int(len(feature_cols_all)),
                "n_features_processed": int(n_comp),
            },
            metric_bundle(X_pca, labels, y_true),
        )

    res_df = pd.DataFrame(results)

    # Ensure optional columns exist across all families for downstream scoring/reporting.
    if "stability_index" not in res_df.columns:
        res_df["stability_index"] = np.nan
    if "noise_ratio" not in res_df.columns:
        res_df["noise_ratio"] = np.nan

    # Ensure core metric columns exist for consistent downstream schema.
    for c in [
        "normalized_mutual_info",
        "adjusted_rand_index",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "stability_index",
    ]:
        if c not in res_df.columns:
            res_df[c] = np.nan

    # Main table order favors external alignment first, then internal quality.
    res_df = res_df.sort_values(["normalized_mutual_info", "adjusted_rand_index", "silhouette"], ascending=False)
    res_df.to_csv(out_csv, index=False)

    # ----------------
    # Create visuals
    # ----------------
    sns.set(style="whitegrid")

    # KMeans k sweep line plots
    for fam, fname in [("kmeans_k_sweep", "kmeans_k_sweep_metrics.png")]:
        sub = res_df[res_df["family"] == fam].copy()
        if len(sub):
            sub = sub.sort_values("k")
            plt.figure(figsize=(8, 4.5))
            plt.plot(sub["k"], sub["normalized_mutual_info"], marker="o", label="NMI")
            plt.plot(sub["k"], sub["adjusted_rand_index"], marker="o", label="ARI")
            plt.plot(sub["k"], sub["silhouette"], marker="o", label="Silhouette")
            plt.title(f"{fam.replace('_', ' ').title()} Metrics vs k")
            plt.xlabel("k")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / fname, dpi=160)
            plt.close()

    # Top model comparison by NMI (alignment-focused view)
    top = res_df.head(12).copy()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="normalized_mutual_info", y="variant", hue="family", dodge=False)
    plt.title("Top Unsupervised Variants by NMI")
    plt.xlabel("Normalized Mutual Information")
    plt.ylabel("Variant")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_variants_by_nmi.png", dpi=160)
    plt.close()

    # Cluster-vs-burnout heatmap for best KMeans-k sweep variant
    # Useful for qualitative interpretation of discovered clusters vs quartile bins.
    kmeans_sweep = res_df[res_df["family"] == "kmeans_k_sweep"]
    if len(kmeans_sweep):
        best_k = int(kmeans_sweep.iloc[0]["k"])
        km_best = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        labels_best = km_best.fit_predict(X_full_proc)
        ct = pd.crosstab(pd.Series(labels_best, name="cluster"), pd.Series(y_true, name="burnout_q_true"))
        plt.figure(figsize=(7, 5))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Best KMeans Sweep: Cluster vs Burnout (k={best_k})")
        plt.xlabel("True Burnout Quartile")
        plt.ylabel("K-Means Cluster")
        plt.tight_layout()
        plt.savefig(fig_dir / "best_kmeans_cluster_vs_burnout.png", dpi=160)
        plt.close()

    # Plain-language metric notes for downstream sharing/reporting.
    metric_explanations = {
        "silhouette": "How clearly separated clusters are (higher is better, range -1 to 1).",
        "davies_bouldin": "Average overlap/similarity between clusters (lower is better).",
        "calinski_harabasz": "Separation between clusters relative to compactness (higher is better).",
        "adjusted_rand_index": "Agreement with true burnout classes, adjusted for chance (higher is better; 1 is perfect).",
        "normalized_mutual_info": "Shared information between clusters and true burnout classes (0 to 1; higher is better).",
        "stability_index": "Average pairwise ARI across repeated runs/seeds of the same setup (higher means more reproducible clusters).",
        "noise_ratio": "Penalty column for consistency; lower is better when present.",
    }
    figure_explanations = {
        "kmeans_k_sweep_metrics.png": "Line chart across k=2..10 showing NMI, ARI, and Silhouette for the KMeans k-sweep family.",
        "top_variants_by_nmi.png": "Bar chart of top model variants ranked by NMI; compares alignment with burnout quartiles.",
        "best_kmeans_cluster_vs_burnout.png": "Heatmap of discovered clusters vs true burnout quartiles for the best kmeans_k_sweep run.",
    }

    # JSON summary bundles top rows + metadata for lightweight downstream consumers.
    summary = {
        "dataset": os.path.relpath(data_path, root),
        "n_rows": int(len(df)),
        "n_features_raw_all": int(len(feature_cols_all)),
        "processed_dimension_all": int(X_full_proc.shape[1]),
        "best_overall": res_df.iloc[0].to_dict() if len(res_df) else {},
        "top_10": res_df.head(10).to_dict(orient="records"),
        "metric_explanations": metric_explanations,
        "figure_explanations": figure_explanations,
        "outputs": {
            "results_csv": os.path.relpath(out_csv, root),
            "results_json": os.path.relpath(out_json, root),
            "figures_dir": os.path.relpath(fig_dir, root),
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("Saved:", os.path.relpath(out_csv, root))
    print("Saved:", os.path.relpath(out_json, root))
    print("Saved figures dir:", os.path.relpath(fig_dir, root))


if __name__ == "__main__":
    main()
