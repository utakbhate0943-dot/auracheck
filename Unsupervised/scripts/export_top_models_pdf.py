"""
Export a PDF report comparing top unsupervised models and metric definitions.

Input:
- Unsupervised/outputs/experiments/unsupervised_experiments_results.csv

Output:
- Unsupervised/outputs/experiments/kmeans_unsupervised_findings_report.pdf
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import textwrap
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_repo_root(start: Path | None = None) -> Path:
    """Find project root by locating the experiment CSV output."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "Unsupervised" / "outputs" / "experiments" / "unsupervised_experiments_results.csv").exists():
            return cand
    raise FileNotFoundError("Could not find repository root with Unsupervised outputs.")


def fmt(x: float | int | str | None, digits: int = 4) -> str:
    """Format values for display in PDF pages/tables."""
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)) or math.isinf(float(x)):
            return "-"
        return f"{float(x):.{digits}f}"
    return str(x)


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Render a simple text-only page and auto-fit line spacing."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=18, fontweight="bold", va="top")

    n_lines = max(len(lines), 1)
    step = min(0.045, (0.88 - 0.06) / n_lines)
    body_fontsize = 11 if n_lines <= 16 else 10

    y = 0.88
    for line in lines:
        ax.text(0.02, y, line, fontsize=body_fontsize, va="top")
        y -= step
        if y < 0.04:
            break

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def wrap_cell_text(value: object, width: int) -> str:
    """Wrap long strings to avoid table-cell overflow."""
    text = str(value)
    if not text or text == "nan":
        return "-"
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def add_table_page(
    pdf: PdfPages,
    title: str,
    table_df: pd.DataFrame,
    notes: list[str] | None = None,
) -> None:
    """Render one table page, with optional metric notes above the table."""
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)

    # Optional explanatory notes that share the same page as the table.
    notes_present = bool(notes)
    if notes_present:
        y = 0.965
        for idx, line in enumerate(notes or []):
            wrapped = textwrap.fill(line, width=120)
            ax.text(
                0.01,
                y,
                wrapped,
                fontsize=9.5 if idx == 0 else 9,
                fontweight="bold" if idx == 0 else "normal",
                va="top",
                transform=ax.transAxes,
            )
            y -= 0.045 if idx == 0 else 0.032

    # Wrap cell contents for readability at larger font sizes.
    display_df = table_df.copy()
    wrap_widths = {
        "family": 10,
        "variant": 18,
        "k": 4,
        "rank": 5,
        "silhouette": 8,
        "davies_bouldin": 8,
        "calinski_harabasz": 8,
        "adjusted_rand_index": 8,
        "normalized_mutual_info": 8,
    }
    for column in display_df.columns:
        display_df[column] = display_df[column].map(
            lambda value, col=column: wrap_cell_text(value, wrap_widths.get(col, 12))
        )

    header_map = {
        "davies_bouldin": "davies\nbouldin",
        "calinski_harabasz": "calinski\nharabasz",
        "adjusted_rand_index": "adjusted rand\nindex",
        "normalized_mutual_info": "normalized\nmutual info",
    }
    display_columns = [header_map.get(column, column) for column in display_df.columns]

    # Manual width profile tuned for this specific metric table layout.
    col_widths = []
    for column in table_df.columns:
        if column == "rank":
            col_widths.append(0.045)
        elif column == "family":
            col_widths.append(0.145)
        elif column == "variant":
            col_widths.append(0.235)
        elif column == "k":
            col_widths.append(0.040)
        else:
            col_widths.append(0.085)

    # Use bbox mode when notes are present so the table does not overlap text.
    table_kwargs: dict[str, object] = {
        "cellText": display_df.values,
        "colLabels": display_columns,
        "cellLoc": "center",
        "colWidths": col_widths,
    }
    if notes_present:
        table_kwargs["bbox"] = [0.0, 0.0, 1.0, 0.72]
    else:
        table_kwargs["loc"] = "center"

    tbl = ax.table(**table_kwargs)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    tbl.scale(1.0, 2.3)

    # Header style
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8EEF8")
            cell.set_height(cell.get_height() * 1.25)
        else:
            cell.set_height(cell.get_height() * 1.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_figure_page(pdf: PdfPages, title: str, image_path: Path, explanation: str) -> None:
    """Render one figure with its plain-language explanation on the same page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Top panel: figure image
    ax_img = fig.add_axes([0.08, 0.24, 0.84, 0.66])
    ax_img.axis("off")

    if image_path.exists():
        img = plt.imread(image_path)
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, f"Figure not found:\n{image_path}", ha="center", va="center", fontsize=11)

    # Bottom panel: explanation text
    ax_txt = fig.add_axes([0.06, 0.05, 0.88, 0.16])
    ax_txt.axis("off")
    ax_txt.text(
        0.0,
        0.95,
        "What this figure shows:",
        fontsize=11,
        fontweight="bold",
        va="top",
        transform=ax_txt.transAxes,
    )
    wrapped = textwrap.fill(explanation, width=145)
    ax_txt.text(0.0, 0.62, wrapped, fontsize=10.5, va="top", transform=ax_txt.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict:
    """Load JSON if present; otherwise return empty dict."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    root = find_repo_root()
    in_csv = root / "Unsupervised" / "outputs" / "experiments" / "unsupervised_experiments_results.csv"
    experiments_json_path = root / "Unsupervised" / "outputs" / "experiments" / "unsupervised_experiments_results.json"
    kmeans_json_path = root / "Unsupervised" / "outputs" / "kmeans" / "kmeans_results.json"
    kmeans_profile_path = root / "Unsupervised" / "outputs" / "kmeans" / "kmeans_cluster_profile_summary.csv"
    kmeans_profile_cat_path = root / "Unsupervised" / "outputs" / "kmeans" / "kmeans_cluster_profile_categorical.csv"
    kmeans_profile_cat_strong_path = (
        root / "Unsupervised" / "outputs" / "kmeans" / "kmeans_cluster_profile_categorical_strong_signals.csv"
    )
    out_pdf = root / "Unsupervised" / "outputs" / "experiments" / "kmeans_unsupervised_findings_report.pdf"

    # Load full experiment output and keep KMeans-only families for this report.
    df = pd.read_csv(in_csv)
    df = df[df["family"].astype(str).str.startswith("kmeans_")].copy()
    selected_families = sorted(df["family"].dropna().unique().tolist())

    # Pick one top variant per family (highest silhouette), then rank families globally.
    family_top = (
        df.sort_values(["family", "silhouette"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .sort_values("silhouette", ascending=False)
        .reset_index(drop=True)
    )

    family_top = family_top.reset_index(drop=True)
    family_top.insert(0, "rank", np.arange(1, len(family_top) + 1))

    display_cols = [
        "rank",
        "family",
        "variant",
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "adjusted_rand_index",
        "normalized_mutual_info",
    ]

    # Table view shown in the PDF.
    top_display = family_top[display_cols].copy()
    top_display["variant"] = top_display["variant"].astype(str).str.slice(0, 38)

    for col in [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "adjusted_rand_index",
        "normalized_mutual_info",
    ]:
        top_display[col] = top_display[col].map(lambda x: fmt(x, 4))

    # Shared metric definitions used in report text/table notes.
    metric_lines = [
        "Metric meanings:",
        "- silhouette: cluster separation (higher better)",
        "- davies_bouldin: cluster overlap (lower better)",
        "- calinski_harabasz (CH): compactness vs separation (higher better)",
        "- adjusted_rand_index (ARI): alignment to burnout quartiles, chance-adjusted (higher better)",
        "- normalized_mutual_info (NMI): shared information with burnout quartiles (higher better)",
        "- stability_index: repeatability across random seeds (higher better)",
        "",
        "Interpretation note:",
        "Compare individual metrics across variants to find the best trade-off for your use case.",
    ]

    # High-level report summary (first page).
    summary_lines = [
        f"Input file: {os.path.relpath(in_csv, root)}",
        f"Rows (model variants): {len(df)}",
        f"Families represented in report: {len(family_top)}",
        f"Included families: {', '.join(selected_families)}",
        "",
        "Top family by silhouette score:",
        f"- family: {family_top.iloc[0]['family']}",
        f"- variant: {family_top.iloc[0]['variant']}",
        f"- silhouette: {fmt(family_top.iloc[0]['silhouette'], 4)}",
    ]

    # Notes intentionally appear on the same page as the table.
    table_metric_notes = [
        "Metric meanings:",
        "• silhouette: cluster separation (higher better)",
        "• davies_bouldin: cluster overlap (lower better)",
        "• calinski_harabasz (CH): compactness vs separation (higher better)",
        "• adjusted_rand_index (ARI): alignment to burnout quartiles (higher better)",
        "• normalized_mutual_info (NMI): shared information with burnout quartiles (higher better)",
    ]

    # Convenience slices for per-family narrative pages.
    top_kmeans_k_sweep = family_top[family_top["family"] == "kmeans_k_sweep"]
    top_kmeans_feature_subset = family_top[family_top["family"] == "kmeans_feature_subset"]
    top_kmeans_pca = family_top[family_top["family"] == "kmeans_pca"]
    top_kmeans_transformer_weight = family_top[family_top["family"] == "kmeans_transformer_weight"]

    def top_value(top_df: pd.DataFrame, column: str, digits: int = 4) -> str:
        """Safely fetch one display value from the top row of a family slice."""
        if len(top_df) == 0:
            return "N/A"
        return fmt(top_df.iloc[0][column], digits)

    family_explanations_page1 = [
        "1. kmeans_k_sweep",
        "Tests k=2 to k=10 on all features (baseline KMeans tuning).",
        "Purpose: find the best cluster count without changing the feature space.",
        "Best model:",
        f"  Variant: {top_value(top_kmeans_k_sweep, 'variant')}",
        f"  Silhouette: {top_value(top_kmeans_k_sweep, 'silhouette')}",
        f"  NMI: {top_value(top_kmeans_k_sweep, 'normalized_mutual_info')}, ARI: {top_value(top_kmeans_k_sweep, 'adjusted_rand_index')}",
        "",
        "2. kmeans_feature_subset",
        "Tests clustering with focused feature domains:",
        "  - all_features: All available features (post-exclusion)",
        "  - psychosocial_focus: Sleep, Physical Activity, Diet, Social Support, Substance Use,",
        "                       Counseling Service Use, Family History, Chronic Illness, Financial Stress",
        "  - academic_lifestyle_focus: Course, CGPA, Semester Credit Load, Extracurricular",
        "                              Involvement, Residence Type, Relationship Status, Age, Gender",
        "Purpose: identify whether specific feature groups separate students better than full features.",
        "Best model:",
        f"  Variant: {top_value(top_kmeans_feature_subset, 'variant')}",
        f"  Silhouette: {top_value(top_kmeans_feature_subset, 'silhouette')}",
        f"  NMI: {top_value(top_kmeans_feature_subset, 'normalized_mutual_info')}, ARI: {top_value(top_kmeans_feature_subset, 'adjusted_rand_index')}",
    ]

    family_explanations_page2 = [
        "3. kmeans_pca",
        "PCA dimensionality reduction followed by KMeans with k=4.",
        "Tests n_components: 2, 5, 10, 20 on all features.",
        "Purpose: reduce high-dimensional one-hot feature space into compact latent components.",
        "Interpretation: lower components simplify structure; higher components retain more detail.",
        "Best model:",
        f"  Variant: {top_value(top_kmeans_pca, 'variant')}",
        f"  Silhouette: {top_value(top_kmeans_pca, 'silhouette')}",
        f"  NMI: {top_value(top_kmeans_pca, 'normalized_mutual_info')}, ARI: {top_value(top_kmeans_pca, 'adjusted_rand_index')}",
        f"  CH: {top_value(top_kmeans_pca, 'calinski_harabasz')}, DB: {top_value(top_kmeans_pca, 'davies_bouldin')}",
        "",
        "4. kmeans_transformer_weight",
        "Tests weighted preprocessing to emphasize numeric vs categorical features.",
        "  - num_heavy (1.5x numeric, 1.0x categorical)",
        "  - cat_heavy (1.0x numeric, 1.5x categorical)",
        "Purpose: test whether burnout patterns are driven more by numeric intensity vs category identity.",
        "Interpretation: num_heavy boosts scaled numeric signals; cat_heavy boosts one-hot category influence.",
        "Best model:",
        f"  Variant: {top_value(top_kmeans_transformer_weight, 'variant')}",
        f"  Silhouette: {top_value(top_kmeans_transformer_weight, 'silhouette')}",
        f"  NMI: {top_value(top_kmeans_transformer_weight, 'normalized_mutual_info')}, ARI: {top_value(top_kmeans_transformer_weight, 'adjusted_rand_index')}",
        f"  CH: {top_value(top_kmeans_transformer_weight, 'calinski_harabasz')}, DB: {top_value(top_kmeans_transformer_weight, 'davies_bouldin')}",
    ]

    # Overall narrative page consolidating what the current unsupervised runs indicate.
    best_row = df.sort_values("silhouette", ascending=False).iloc[0] if len(df) else None
    insights_lines = [
        "Overall model performance summary:",
        "- Clustering finds structure, but alignment to burnout quartiles is weak (ARI/NMI near zero across families).",
        "- This means clusters are useful for segmentation, not strong burnout prediction as-is.",
        "- PCA-based variants (especially low components) improve geometric separation the most.",
        "- kmeans_k_sweep runs are highly stable across seeds for many k values.",
        "",
        "Recommended interpretation:",
        "- Use these models for student profile discovery and intervention design by segment.",
        "- Do not use cluster labels alone as a clinical/operational burnout predictor.",
    ]
    if best_row is not None:
        insights_lines.extend(
            [
                "",
                "Best geometric variant observed:",
                f"- family: {best_row['family']}",
                f"- variant: {best_row['variant']}",
                f"- silhouette: {fmt(best_row['silhouette'])}",
                f"- davies_bouldin: {fmt(best_row['davies_bouldin'])}",
            ]
        )

    # Load payloads used by interpretation and figure pages.
    experiments_payload = load_json(experiments_json_path)
    kmeans_payload = load_json(kmeans_json_path)

    # Dedicated interpretation page: what cluster personas look like and how to justify them.
    cluster_interp_lines = [
        "How to interpret KMeans clusters (personas):",
        "- We distinguish clusters by comparing per-cluster feature means vs overall means.",
        "- Numeric means come from kmeans_cluster_profile_summary.csv (Age, CGPA, Financial_Stress, Credit_Load).",
        "- We verify structure using silhouette/DB/CH and visualize overlap via PCA and heatmaps.",
    ]

    if kmeans_profile_path.exists():
        prof = pd.read_csv(kmeans_profile_path)
        use_cols = [c for c in ["Age", "CGPA", "Financial_Stress", "Semester_Credit_Load"] if c in prof.columns]
        if len(prof) and use_cols:
            overall = {c: float(prof[c].mean()) for c in use_cols}
            cluster_interp_lines.extend(["", "Cluster personas from profile means:"])
            for _, row in prof.sort_values("cluster").iterrows():
                tags: list[str] = []
                for col, label in [
                    ("Financial_Stress", "financial stress"),
                    ("Semester_Credit_Load", "credit load"),
                    ("Age", "age"),
                    ("CGPA", "CGPA"),
                ]:
                    if col not in overall or col not in row:
                        continue
                    diff = float(row[col]) - overall[col]
                    if abs(diff) < 0.10:
                        tags.append(f"average {label}")
                    elif diff > 0:
                        tags.append(f"higher {label}")
                    else:
                        tags.append(f"lower {label}")

                cluster_interp_lines.append(
                    f"- Cluster {int(row['cluster'])}: " + (", ".join(tags) if tags else "mixed profile")
                )

    km_metrics = (kmeans_payload.get("metrics_unsupervised") or {})
    km_align = (kmeans_payload.get("alignment_to_burnout_labels") or {})
    if km_metrics or km_align:
        cluster_interp_lines.extend(
            [
                "",
                "Why these are clusters (and limits):",
                f"- silhouette={fmt(km_metrics.get('silhouette'))}, davies_bouldin={fmt(km_metrics.get('davies_bouldin'))}, calinski_harabasz={fmt(km_metrics.get('calinski_harabasz'))}",
                f"- ARI={fmt(km_align.get('adjusted_rand_index'))}, NMI={fmt(km_align.get('normalized_mutual_info'))} vs burnout quartiles.",
                "- Conclusion: useful segmentation personas, but weak direct burnout-label alignment.",
            ]
        )

    # Categorical findings: strongest category per feature/cluster and whether any strong lifts exist.
    if kmeans_profile_cat_path.exists():
        cat_prof = pd.read_csv(kmeans_profile_cat_path)
        cluster_interp_lines.extend(["", "Categorical-feature findings:"])

        if kmeans_profile_cat_strong_path.exists():
            cat_strong = pd.read_csv(kmeans_profile_cat_strong_path)
            if len(cat_strong) == 0:
                cluster_interp_lines.append(
                    "- No strong categorical signals met thresholds (cluster_share >= 0.15 and lift >= 1.20)."
                )
                cluster_interp_lines.append(
                    "- Interpretation: categories vary by cluster only modestly; numeric features are the main differentiators."
                )
            else:
                cluster_interp_lines.append(
                    f"- Strong categorical signals found: {len(cat_strong)} rows above threshold."
                )

        # Add compact examples from top-lift rows for interpretability.
        if len(cat_prof):
            top_cat = cat_prof.sort_values("lift", ascending=False).head(4)
            for _, r in top_cat.iterrows():
                cluster_interp_lines.append(
                    f"- Cluster {int(r['cluster'])}: {r['feature']}={r['top_category']} "
                    f"({fmt(float(r['cluster_share']) * 100, 1)}% vs {fmt(float(r['overall_share']) * 100, 1)}%, "
                    f"lift={fmt(r['lift'], 2)})"
                )

    # PCA component interpretation page from current KMeans preprocessing.
    pca_lines = [
        "PC1/PC2 meaning in this project:",
    ]
    data_path = root / "Dataset" / "students_mental_health_survey_with_burnout_final.csv"
    if data_path.exists():
        raw_df = pd.read_csv(data_path)
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
        feature_cols = [c for c in raw_df.columns if c not in excluded]
        X = raw_df[feature_cols].copy()
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
        )
        X_proc = preprocess.fit_transform(X)

        cat_feature_names = preprocess.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_feature_names

        pca_model = SKPCA(n_components=2, random_state=42)
        pca_model.fit(X_proc)
        loadings = pd.DataFrame(
            {
                "feature": feature_names,
                "PC1": pca_model.components_[0],
                "PC2": pca_model.components_[1],
            }
        )

        top_pc1 = (
            loadings.assign(abs_pc1=loadings["PC1"].abs())
            .sort_values("abs_pc1", ascending=False)
            .head(4)
        )
        top_pc2 = (
            loadings.assign(abs_pc2=loadings["PC2"].abs())
            .sort_values("abs_pc2", ascending=False)
            .head(4)
        )

        pca_lines.extend(
            [
                f"- Explained variance: PC1={fmt(float(pca_model.explained_variance_ratio_[0]), 4)}, PC2={fmt(float(pca_model.explained_variance_ratio_[1]), 4)}",
                "",
                "Top weighted features in PC1:",
            ]
        )
        for _, r in top_pc1.iterrows():
            pca_lines.append(f"- {r['feature']}: {fmt(float(r['PC1']), 3)}")

        pca_lines.extend(["", "Top weighted features in PC2:"])
        for _, r in top_pc2.iterrows():
            pca_lines.append(f"- {r['feature']}: {fmt(float(r['PC2']), 3)}")

        pca_lines.extend(
            [
                "",
                "Interpretation:",
                "- PC1/PC2 are weighted combinations of original processed features, not raw columns.",
                "- They summarize dominant variation for visualization and can highlight which variables drive separation.",
            ]
        )

    # Figure sources + explanations from both pipelines.

    figure_pages: list[tuple[str, Path, str]] = []

    exp_fig_dir = root / "Unsupervised" / "outputs" / "experiments" / "figures"
    for fname, text in (experiments_payload.get("figure_explanations") or {}).items():
        figure_pages.append((f"Experiments Figure: {fname}", exp_fig_dir / fname, str(text)))

    km_outputs = (kmeans_payload.get("outputs") or {}).get("figures") or {}
    km_explanations = kmeans_payload.get("figure_explanations") or {}
    # Prefer explicit paths from JSON output mapping.
    for label, rel_path in km_outputs.items():
        fig_path = root / str(rel_path)
        fname = Path(str(rel_path)).name
        exp = str(km_explanations.get(fname, f"Figure for {label}."))
        figure_pages.append((f"KMeans Figure: {fname}", fig_path, exp))

    with PdfPages(out_pdf) as pdf:
        add_text_page(pdf, "Top KMeans Models — Comparison Report", summary_lines)
        add_text_page(pdf, "KMeans Family Descriptions (1/2)", family_explanations_page1)
        add_text_page(pdf, "KMeans Family Descriptions (2/2)", family_explanations_page2)
        add_text_page(pdf, "Final Unsupervised Insights", insights_lines)
        add_text_page(pdf, "KMeans Cluster Interpretation", cluster_interp_lines)
        add_text_page(pdf, "PCA Component Interpretation (PC1/PC2)", pca_lines)
        add_table_page(
            pdf,
            f"Top Models by Silhouette Score ({len(top_display)} Families)",
            top_display,
            notes=table_metric_notes,
        )
        for title, image_path, explanation in figure_pages:
            add_figure_page(pdf, title, image_path, explanation)

    print("Saved:", os.path.relpath(out_pdf, root))


if __name__ == "__main__":
    main()
