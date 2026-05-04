"""AuraCheck Streamlit application.

This module intentionally keeps UI rendering, authentication, local persistence,
and optional Supabase sync together to simplify project delivery.
"""


import os
import json
import uuid
import secrets
import hashlib
import hmac
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from supabase import create_client

load_dotenv(override=True)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "utakbhate0943@sdsu.com").strip().lower()
TEST_DEMO_EMAIL = "test@gmail.com"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
USER_RESPONSES_JSON_PATH = os.path.join(APP_DIR, "Dataset", "user_responses.json")
BASELINE_OUTPUT_DIR = os.path.join(APP_DIR, "baseline", "outputs", "final_baseline_model")
BASELINE_METRICS_PATH = os.path.join(BASELINE_OUTPUT_DIR, "production_pruned_multinomial_metrics.csv")
BASELINE_CM_PATH = os.path.join(BASELINE_OUTPUT_DIR, "final_selected_baseline_confusion_matrix.csv")
BASELINE_SENS_SPEC_PATH = os.path.join(BASELINE_OUTPUT_DIR, "final_selected_baseline_sensitivity_specificity.csv")
RF_SUMMARY_PATH = os.path.join(APP_DIR, "ml_randomforest", "outputs", "random_forest_outputs_summary.json")
XGB_MODEL2_METRICS_PATH = os.path.join(APP_DIR, "xgboost-model", "output", "model_tuned", "xgboost_metrics.csv")
XGB_MODEL2_CM_PATH = os.path.join(APP_DIR, "xgboost-model", "output", "model_tuned", "xgboost_confusion_matrix_tuned.csv")
KMEANS_PCA2_MODEL_DIR = os.path.join(APP_DIR, "Unsupervised", "outputs", "pca2_kmeans_model")
DATASET_FINAL_PATH = os.path.join(APP_DIR, "Dataset", "students_mental_health_survey_with_burnout_final.csv")

REQUIRED_FIELDS = ["Age", "Course", "Gender", "CGPA", "Sleep_Quality", "Physical_Activity", "Diet_Quality", "Social_Support", "Relationship", "Substance_Use", "Counseling", "Family_History", "Chronic_Illness", "Financial_Stress", "Extracurricular", "Semester", "Residence_Type"]
POSITIVE_THOUGHTS = ["🌟 You are capable of overcoming challenges", "💚 Your mental health matters and deserves attention", "🌈 Every day is a fresh opportunity for growth", "💫 You have strength within you", "🌸 Self-care is not selfish, it's essential", "⭐ Progress over perfection always", "🎯 Your feelings are valid and important", "🌊 Challenges help you grow stronger", "💡 You deserve to be happy and healthy", "🦋 Transformation starts with self-compassion"]
STATIC_USER_FIELDS = {
    "Age": "survey_age",
    "Course": "survey_course",
    "Gender": "survey_gender",
    "CGPA": "survey_cgpa",
    "Relationship": "survey_relationship",
    "Family_History": "survey_family_history",
    "Semester": "survey_semester",
    "Residence_Type": "survey_residence_type",
}

def normalize_supabase_url(url_value: str) -> str:
    """Normalize Supabase URL so env variants do not break client setup."""
    normalized = (url_value or "").strip().strip('"').strip("'").rstrip("/")
    if normalized and not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    return normalized

SUPABASE_URL = normalize_supabase_url(os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or "").strip()
st.set_page_config(page_title="AuraCheck", page_icon="💜", layout="wide")

st.markdown("""
<style>
</style>
""", unsafe_allow_html=True)


def load_json_file(path: str, default: Optional[dict] = None) -> dict:
    """Load a JSON file with a safe fallback."""
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def load_kmeans_pca2_metrics() -> dict:
    """Load PCA2 KMeans artifacts and compute alignment metrics for comparison."""
    model_dir = Path(KMEANS_PCA2_MODEL_DIR)
    preprocessor_path = model_dir / "pca2_kmeans_preprocessor.joblib"
    pca_path = model_dir / "pca2_kmeans_pca.joblib"
    model_path = model_dir / "pca2_kmeans_model.joblib"
    metadata_path = model_dir / "pca2_kmeans_metadata.json"

    required_paths = [preprocessor_path, pca_path, model_path, metadata_path, Path(DATASET_FINAL_PATH)]
    if any(not p.exists() for p in required_paths):
        return {}

    try:
        preprocessor = joblib.load(preprocessor_path)
        pca = joblib.load(pca_path)
        kmeans = joblib.load(model_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        df = pd.read_csv(DATASET_FINAL_PATH)
        feature_cols = metadata.get("features_raw", [])
        if not feature_cols or any(c not in df.columns for c in feature_cols):
            return {}
        if "burnout_raw_score" not in df.columns:
            return {}

        X = df[feature_cols].copy()
        num_cols = [c for c in metadata.get("numeric_features", []) if c in X.columns]
        cat_cols = [c for c in metadata.get("categorical_features", []) if c in X.columns]

        for c in num_cols:
            if X[c].isnull().any():
                X[c] = X[c].fillna(X[c].median())
        for c in cat_cols:
            if X[c].isnull().any():
                X[c] = X[c].fillna("Unknown")

        X_proc = preprocessor.transform(X)
        X_pca = pca.transform(X_proc)
        clusters = np.asarray(kmeans.predict(X_pca), dtype=int)

        y_true = pd.qcut(
            df["burnout_raw_score"].astype(float),
            q=4,
            labels=[0, 1, 2, 3],
            duplicates="drop",
        ).astype(int).to_numpy()

        mapping_df = pd.DataFrame({"cluster": clusters, "burnout_q": y_true})
        cluster_to_class = mapping_df.groupby("cluster")["burnout_q"].agg(lambda s: int(s.value_counts().idxmax())).to_dict()
        mapped_classes = np.asarray([cluster_to_class.get(int(c), 1) for c in clusters], dtype=int)

        silhouette = 0.0
        if len(np.unique(clusters)) > 1 and len(clusters) > len(np.unique(clusters)):
            silhouette = float(silhouette_score(X_pca, clusters))

        return {
            "model_type": metadata.get("model_type", "PCA(2) + KMeans(4)"),
            "normalized_mutual_info": float(normalized_mutual_info_score(y_true, mapped_classes)),
            "adjusted_rand_index": float(adjusted_rand_score(y_true, mapped_classes)),
            "silhouette": silhouette,
            "n_clusters": int(metadata.get("n_clusters", len(np.unique(clusters)))),
            "source": "predict_pca2_kmeans.py artifacts",
        }
    except Exception:
        return {}


def load_model_analysis_data() -> dict:
    """Load comparison data for the model analysis page."""
    baseline_metrics = {}
    baseline_cm = pd.DataFrame()
    # baseline_sens_spec = pd.DataFrame()
    xgb_metrics = {}
    xgb_cm = pd.DataFrame()

    if os.path.exists(BASELINE_METRICS_PATH):
        baseline_metrics_df = pd.read_csv(BASELINE_METRICS_PATH)
        if not baseline_metrics_df.empty:
            baseline_metrics = baseline_metrics_df.iloc[0].to_dict()

    if os.path.exists(BASELINE_CM_PATH):
        baseline_cm = pd.read_csv(BASELINE_CM_PATH)

    # if os.path.exists(BASELINE_SENS_SPEC_PATH):
    #     baseline_sens_spec = pd.read_csv(BASELINE_SENS_SPEC_PATH)

    rf_summary = load_json_file(RF_SUMMARY_PATH, {})

    if os.path.exists(XGB_MODEL2_METRICS_PATH):
        xgb_metrics_df = pd.read_csv(XGB_MODEL2_METRICS_PATH)
        if not xgb_metrics_df.empty and {"Metric", "Value"}.issubset(xgb_metrics_df.columns):
            xgb_metrics = {
                str(row["Metric"]).strip(): float(row["Value"])
                for _, row in xgb_metrics_df.iterrows()
            }

    if os.path.exists(XGB_MODEL2_CM_PATH):
        xgb_cm = pd.read_csv(XGB_MODEL2_CM_PATH, index_col=0)

    kmeans_pca2 = load_kmeans_pca2_metrics()

    return {
        "baseline": {
            "metrics": baseline_metrics,
            "confusion_matrix": baseline_cm,
            # "sensitivity_specificity": baseline_sens_spec,
        },
        "random_forest": rf_summary,
        "xgboost": {
            "metrics": xgb_metrics,
            "confusion_matrix": xgb_cm,
        },
        "kmeans_pca2": kmeans_pca2,
    }


def build_model_comparison_table(analysis_data: dict) -> pd.DataFrame:
    """Build a compact comparison table for the model families."""
    baseline_metrics = analysis_data.get("baseline", {}).get("metrics", {})
    rf_summary = analysis_data.get("random_forest", {})
    rf_eval = rf_summary.get("evaluation_matrix", {}) if isinstance(rf_summary, dict) else {}
    rf_output = rf_summary.get("output", {}) if isinstance(rf_summary, dict) else {}
    xgb_metrics = analysis_data.get("xgboost", {}).get("metrics", {})
    kmeans_pca2 = analysis_data.get("kmeans_pca2", {}) if isinstance(analysis_data.get("kmeans_pca2", {}), dict) else {}

    rows = [
        {
            "Model": "Baseline multinomial logistic regression",
            "Type": "Supervised baseline",
            "Primary metric": float(baseline_metrics.get("Accuracy", 0.0)),
            "Secondary metric": float(baseline_metrics.get("Macro_Recall", 0.0)),
            "Notes": "Pruned 14-feature logistic regression; best for interpretability, weakest predictive lift.",
        },
        {
            "Model": "Random forest (SMOTE + tuned trees)",
            "Type": "Supervised candidate",
            "Primary metric": float(rf_output.get("accuracy", 0.0)),
            "Secondary metric": float(rf_eval.get("macro avg", {}).get("recall", 0.0)),
            "Notes": "Best supervised accuracy in this project; non-linear model captures more structure.",
        },
        {
            "Model": "XGBoost (tuned model_tuned)",
            "Type": "Supervised candidate",
            "Primary metric": float(xgb_metrics.get("Accuracy", 0.0)),
            "Secondary metric": float(xgb_metrics.get("Macro Recall", 0.0)),
            "Notes": "Gradient boosting benchmark from xgboost-model outputs; evaluated from tuned model_tuned artifacts.",
        },
        {
            "Model": "KMeans PCA2 (trained model)",
            "Type": "Unsupervised segmentation",
            "Primary metric": float(kmeans_pca2.get("normalized_mutual_info", 0.0)),
            "Secondary metric": float(kmeans_pca2.get("silhouette", 0.0)),
            "Notes": "Computed from PCA(2)+KMeans trained artifacts used by predict_pca2_kmeans.py.",
        },
    ]
    return pd.DataFrame(rows)


def style_eval_table(df: pd.DataFrame, *, hide_index: bool = False, integer_values: bool = False):
    """Return a centered, high-contrast table style for evaluation sections."""
    if df is None or df.empty:
        return df

    def _format_value(value: Any) -> Any:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value:d}" if integer_values else f"{value:.4f}"
        if isinstance(value, float):
            return f"{value:.0f}" if integer_values else f"{value:.4f}"
        return value

    styled = (
        df.style
        .format(_format_value, na_rep="-")
        .set_properties(subset=pd.IndexSlice[:, :], **{
            "text-align": "center",
            "font-weight": "700",
            "color": "#1E1633",
            "background-color": "#E7DEEF",
        })
        .set_table_styles([
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #BBAFD1"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "800"),
                    ("color", "#1B1330"),
                    ("background-color", "#D3C4E6"),
                    ("border", "1px solid #B9A9CF"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "700"),
                    ("color", "#1E1633"),
                    ("border", "1px solid #C5B7D8"),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "800"),
                    ("color", "#1B1330"),
                    ("background-color", "#DDD1EB"),
                    ("border", "1px solid #B9A9CF"),
                ],
            },
        ])
    )

    if hide_index:
        try:
            styled = styled.hide(axis="index")
        except Exception:
            pass

    return styled


def render_model_analysis_page() -> None:
    """Render the model comparison and recommendation page."""
    analysis_data = load_model_analysis_data()
    baseline_metrics = analysis_data.get("baseline", {}).get("metrics", {})
    baseline_cm = analysis_data.get("baseline", {}).get("confusion_matrix", pd.DataFrame())
    # baseline_sens_spec = analysis_data.get("baseline", {}).get("sensitivity_specificity", pd.DataFrame())
    rf_summary = analysis_data.get("random_forest", {}) if isinstance(analysis_data.get("random_forest", {}), dict) else {}
    rf_eval = rf_summary.get("evaluation_matrix", {})
    rf_cm = rf_summary.get("confusion_matrix", {})
    xgb_metrics = analysis_data.get("xgboost", {}).get("metrics", {})
    xgb_cm = analysis_data.get("xgboost", {}).get("confusion_matrix", pd.DataFrame())
    kmeans_pca2 = analysis_data.get("kmeans_pca2", {}) if isinstance(analysis_data.get("kmeans_pca2", {}), dict) else {}

    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            .analysis-hero {
                background: linear-gradient(120deg, rgba(53,93,203,0.16), rgba(155,127,181,0.22));
                border: 2px solid rgba(53,93,203,0.25);
                border-radius: 18px;
                padding: 18px 20px;
                margin: 8px 6px 14px;
            }
            .analysis-title {
                color: #2A1940;
                font-size: 30px;
                font-weight: 800;
                letter-spacing: 0.2px;
                margin: 0 0 6px 0;
            }
            .analysis-subtitle {
                color: #3C2A57;
                font-size: 14px;
                font-weight: 600;
                margin: 0;
            }
            .analysis-pill-row {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 10px;
            }
            .analysis-pill {
                border-radius: 999px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 700;
            }
            .analysis-pill-good { background: #E8F7EC; color: #1A6A37; border: 1px solid #9EDAB3; }
            .analysis-pill-mid { background: #FFF6DD; color: #8A5A00; border: 1px solid #E6C778; }
            .analysis-pill-risk { background: #FFE8EA; color: #8A1F2C; border: 1px solid #E7A9B1; }
            .analysis-note {
                border-radius: 14px;
                padding: 12px 14px;
                margin: 8px 0;
                font-size: 13px;
                font-weight: 600;
            }
            .analysis-note-good { background: #EAF8EE; color: #195C34; border: 1px solid #9FD8B3; }
            .analysis-note-warn { background: #FFF7E5; color: #7C5500; border: 1px solid #E7C46B; }
            .analysis-note-risk { background: #FFEDEF; color: #7C1F2A; border: 1px solid #E6A8B0; }
            .metric-card {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid #BCAED3;
                background: #EEE6F5;
                min-height: 156px;
            }
            .metric-card-head {
                padding: 10px 12px;
                text-align: center;
                font-size: 14px;
                font-weight: 800;
                color: #F7F2FF;
                background: #4A3A66;
                letter-spacing: 0.2px;
            }
            .metric-card-body {
                padding: 14px 12px 12px;
                text-align: center;
            }
            .metric-card-value {
                font-size: 44px;
                line-height: 1;
                font-weight: 900;
                color: #2A1940;
                margin: 0 0 10px 0;
            }
            .metric-card-caption {
                font-size: 13px;
                font-weight: 700;
                color: #5A4E73;
                margin: 0;
            }
            .metric-card-rf {
                border: 2px solid #79C798;
                background: #E5F5EA;
                box-shadow: 0 6px 18px rgba(46, 139, 87, 0.18);
            }
            .metric-card-rf .metric-card-head {
                color: #F2FFF7;
                background: #1F7A46;
            }
            .metric-card-rf .metric-card-value {
                color: #185B34;
            }
            .metric-badge {
                display: inline-block;
                margin-top: 8px;
                padding: 3px 10px;
                border-radius: 999px;
                font-size: 11px;
                font-weight: 800;
                color: #0F512B;
                background: #BFE8CB;
                border: 1px solid #7DC595;
            }
        </style>
        <div class='analysis-hero'>
            <p class='analysis-title'>Model Analysis and Recommendation</p>
            <p class='analysis-subtitle'>Baseline logistic regression, random forest, and XGBoost are compared with confusion matrices; KMeans is shown as an unsupervised alignment check.</p>
            <div class='analysis-pill-row'>
                <span class='analysis-pill analysis-pill-good'>Prediction Winner: Random Forest</span>
                <span class='analysis-pill analysis-pill-mid'>Supervised Benchmark: XGBoost</span>
                <span class='analysis-pill analysis-pill-mid'>Benchmark Model: Baseline</span>
                <span class='analysis-pill analysis-pill-risk'>Exploration Only: KMeans</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Back to AuraCheck", key="analysis_back_to_main", width="stretch"):
        st.session_state["auth_page"] = "main"
        st.rerun()

    baseline_accuracy_text = f"{float(baseline_metrics.get('Accuracy', 0.0)):.1%}"
    rf_accuracy_text = f"{float(rf_summary.get('output', {}).get('accuracy', 0.0)):.1%}"
    xgb_accuracy_text = f"{float(xgb_metrics.get('Accuracy', 0.0)):.1%}"
    kmeans_silhouette_text = f"{float(kmeans_pca2.get('silhouette', 0.0)):.4f}"

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-card-head'>Baseline Accuracy</div>
                <div class='metric-card-body'>
                    <p class='metric-card-value'>{baseline_accuracy_text}</p>
                    <p class='metric-card-caption'>Pruned multinomial logistic regression</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class='metric-card metric-card-rf'>
                <div class='metric-card-head'>Random Forest Accuracy</div>
                <div class='metric-card-body'>
                    <p class='metric-card-value'>{rf_accuracy_text}</p>
                    <p class='metric-card-caption'>SMOTE + 300-tree random forest</p>
                    <span class='metric-badge'>Ideal Model</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-card-head'>XGBoost Accuracy</div>
                <div class='metric-card-body'>
                    <p class='metric-card-value'>{xgb_accuracy_text}</p>
                    <p class='metric-card-caption'>Tuned XGBoost (model_tuned)</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-card-head'>KMeans (PCA2) Metrics</div>
                <div class='metric-card-body'>
                    <p class='metric-card-value'>{kmeans_silhouette_text}</p>
                    <p class='metric-card-caption'>Silhouette score</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### How each model behaves")
    insight_cols = st.columns(4)
    with insight_cols[0]:
        st.markdown(
            """
            <div class='card-style' style='border-left: 6px solid #C39A3A; padding: 18px;'>
            <strong style='font-size:18px;'>Baseline model</strong><br/>
            The baseline script uses a pruned multinomial logistic regression pipeline. It is easy to explain,
            but the saved metrics show <strong>weaker predictive power</strong>, so it is best treated as a reference point.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[1]:
        st.markdown(
            """
            <div class='card-style' style='border-left: 6px solid #2E8B57; padding: 18px;'>
            <strong style='font-size:18px;'>Random Forest</strong><br/>
            The random forest script adds non-linearity and uses SMOTE on the training split. That gave the
            <strong>strongest supervised performance</strong> in this project and the most balanced class recovery overall.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[2]:
        st.markdown(
            """
            <div class='card-style' style='border-left: 6px solid #246A9A; padding: 18px;'>
            <strong style='font-size:18px;'>XGBoost</strong><br/>
            The tuned XGBoost run provides a gradient-boosted tree baseline for comparison. It captures non-linear effects,
            but the saved tuned metrics are <strong>still below random forest</strong> in this repository outputs.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[3]:
        st.markdown(
            """
            <div class='card-style' style='border-left: 6px solid #A04857; padding: 18px;'>
            <strong style='font-size:18px;'>KMeans</strong><br/>
            KMeans is an exploratory segmentation tool, not a predictive classifier. Its internal clustering
            metrics are modest and its alignment with burnout labels is <strong>near zero</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Performance comparison")
    st.caption("Supervised models are compared directly by prediction metrics; KMeans is shown separately as a clustering check.")
    comparison_df = build_model_comparison_table(analysis_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("### Evaluation details")
    st.caption("Baseline, Random Forest, and XGBoost include confusion-matrix views. KMeans does not produce a standard confusion matrix, so its clustering quality is shown with NMI, ARI, and silhouette.")
    eval_cols = st.columns(3)
    with eval_cols[0]:
        st.markdown("<h4 style='text-align:center;'>Baseline confusion metrics</h4>", unsafe_allow_html=True)
        st.table(style_eval_table(pd.DataFrame([baseline_metrics]), hide_index=True))
        # if not baseline_sens_spec.empty:
        #     st.dataframe(baseline_sens_spec, use_container_width=True, hide_index=True)
        if not baseline_cm.empty:
            st.table(style_eval_table(baseline_cm, integer_values=True))
            baseline_heatmap = go.Figure(
                data=go.Heatmap(
                    z=baseline_cm.iloc[:, 1:].values if baseline_cm.shape[1] > 4 else baseline_cm.values,
                    x=list(baseline_cm.columns[1:]) if baseline_cm.shape[1] > 4 else list(baseline_cm.columns),
                    y=list(baseline_cm.iloc[:, 0]) if baseline_cm.shape[1] > 4 else list(baseline_cm.index),
                    colorscale="YlOrBr",
                    showscale=True,
                )
            )
            baseline_heatmap.update_layout(
                title="Baseline Confusion Matrix (Amber benchmark)",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(baseline_heatmap, use_container_width=True)
    with eval_cols[1]:
        st.markdown("<h4 style='text-align:center;'>Random Forest confusion metrics</h4>", unsafe_allow_html=True)
        rf_metrics_table = pd.DataFrame([{
            "accuracy": float(rf_summary.get("output", {}).get("accuracy", 0.0)),
            "macro_precision": float(rf_eval.get("macro avg", {}).get("precision", 0.0)),
            "macro_recall": float(rf_eval.get("macro avg", {}).get("recall", 0.0)),
            "macro_f1": float(rf_eval.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(rf_eval.get("weighted avg", {}).get("f1-score", 0.0)),
            "total_predictions": int(rf_summary.get("output", {}).get("total_predictions", 0)),
        }])
        st.table(style_eval_table(rf_metrics_table, hide_index=True))
        rf_cm_data = rf_cm.get("matrix", []) if isinstance(rf_cm, dict) else []
        rf_labels = rf_cm.get("labels", []) if isinstance(rf_cm, dict) else []
        if rf_cm_data:
            st.table(style_eval_table(pd.DataFrame(rf_cm_data, index=rf_labels, columns=rf_labels), integer_values=True))
            # Add spacing so the RF heatmap aligns better with adjacent model charts.
            st.markdown("<div style='height: 44px;'></div>", unsafe_allow_html=True)
            rf_heatmap = go.Figure(
                data=go.Heatmap(
                    z=rf_cm_data,
                    x=rf_labels,
                    y=rf_labels,
                    colorscale="Greens",
                    showscale=True,
                )
            )
            rf_heatmap.update_layout(
                title="Random Forest Confusion Matrix (Green winner)",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(rf_heatmap, use_container_width=True)
    with eval_cols[2]:
        st.markdown("<h4 style='text-align:center;'>XGBoost confusion metrics</h4>", unsafe_allow_html=True)
        xgb_metrics_table = pd.DataFrame([{
            "accuracy": float(xgb_metrics.get("Accuracy", 0.0)),
            "macro_recall": float(xgb_metrics.get("Macro Recall", 0.0)),
            "macro_f1": float(xgb_metrics.get("Macro F1", 0.0)),
            "kappa": float(xgb_metrics.get("Kappa", 0.0)),
            "roc_auc_ovr": float(xgb_metrics.get("ROC-AUC OvR", 0.0)),
            "log_loss": float(xgb_metrics.get("Log Loss", 0.0)),
        }])
        st.table(style_eval_table(xgb_metrics_table, hide_index=True))

        if not xgb_cm.empty:
            st.table(style_eval_table(xgb_cm, integer_values=True))
            xgb_heatmap = go.Figure(
                data=go.Heatmap(
                    z=xgb_cm.values,
                    x=list(xgb_cm.columns),
                    y=list(xgb_cm.index),
                    colorscale="Blues",
                    showscale=True,
                )
            )
            xgb_heatmap.update_layout(
                title="XGBoost Tuned Confusion Matrix (Blue benchmark)",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(xgb_heatmap, use_container_width=True)

    st.markdown("### Recommendation")
    st.markdown(
        """
        <div style='background: linear-gradient(120deg, #EAF8EE, #F5FFF8); border: 2px solid #9ED8B3; border-radius: 16px; padding: 16px 18px; margin-top: 4px;'>
            <div style='font-size:18px; font-weight:800; color:#1C5C36; margin-bottom:8px;'>Recommended Production Model: Random Forest</div>
            <div style='font-size:14px; font-weight:600; color:#2C3A3A; line-height:1.6;'>
                Random Forest is the best choice for final predictive use because it gives the <strong>highest supervised accuracy</strong> and
                <strong>better macro-level recall</strong> than the baseline and XGBoost outputs in this repo. Keep baseline logistic regression and XGBoost for comparison,
                and keep KMeans for exploratory segmentation only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

def get_question_for_field(field_name: str) -> str:
    """Get conversational question for a given field."""
    questions = {
        "Age": "🎂 What's your age?",
        "Course": "📚 What's your course/major?",
        "Gender": "👤 How do you identify?",
        "CGPA": "📊 What's your current CGPA?",
        "Sleep_Quality": "😴 How's your sleep quality?",
        "Physical_Activity": "🏃 How active are you physically?",
        "Diet_Quality": "🥗 How's your diet quality?",
        "Social_Support": "👥 Do you have good social support?",
        "Relationship": "❤️ How's your relationship status?",
        "Substance_Use": "🚭 Do you use substances (caffeine, etc)?",
        "Counseling": "🗣️ Are you in counseling?",
        "Family_History": "👨‍👩‍👧 Any family history of mental health issues?",
        "Chronic_Illness": "🏥 Do you have any chronic illness?",
        "Financial_Stress": "💰 Do you experience financial stress?",
        "Extracurricular": "🎨 Are you involved in extracurricular activities?",
        "Semester": "📅 how many semesters have you enrolled in?",
        "Residence_Type": "🏠 Where do you stay?",
    }
    return questions.get(field_name, f"Tell me about {field_name}")

def get_field_options(field_name: str) -> list:
    """Get predefined button options for each field."""
    options = {
        "Age": [],
        "Gender": ["🧑 Male", "👩 Female", "🧑‍🤝‍🧑 Other"],
        "Course": ["Medical", "Business", "Engineering", "Law", "Computer", "Others"],
        "CGPA": [],
        "Sleep_Quality": ["Good", "Poor", "Average"],
        "Physical_Activity": ["Low", "Moderate", "High"],
        "Diet_Quality": ["Poor", "Average", "Good"],
        "Social_Support": ["Low", "Moderate", "High"],
        "Relationship": ["Single", "Married", "In a relationship"],
        "Substance_Use": ["Never", "Occasionally", "Frequently"],
        "Counseling": ["Never", "Occasionally", "Frequently"],
        "Family_History": ["Yes", "No"],
        "Chronic_Illness": ["Yes", "No"],
        "Financial_Stress": ["🔴 Very High (5)", "🟠 High (4)", "🟡 Moderate (3)", "🟢 Low (2)", "💚 None (1)"],
        "Extracurricular": ["Low", "Moderate", "High"],
        "Semester": ["15", "16", "17", "18", "19","20", "21","22","23","24","25","26","27","28","29","30","30+"],
        "Residence_Type": ["on-campus", "off-campus","With Family"],
    }
    return options.get(field_name, [])


def _parse_numeric_answer(value: Any, default: float) -> float:
    """Parse a numeric answer from numbers, strings, or legacy range labels."""
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value or "").strip()
    if not text:
        return float(default)

    matches = re.findall(r"\d+(?:\.\d+)?", text)
    if not matches:
        return float(default)

    numbers = [float(match) for match in matches]
    if "-" in text and len(numbers) >= 2:
        return float(sum(numbers[:2]) / 2.0)
    return float(numbers[0])


def _format_age_value(value: Any) -> int:
    """Return a safe integer age for UI defaults and storage."""
    return int(round(_parse_numeric_answer(value, 21.0)))


def _format_cgpa_value(value: Any) -> float:
    """Return a safe CGPA value for UI defaults and storage."""
    return round(_parse_numeric_answer(value, 3.0), 2)


# Persistence layer (Supabase-only)
def get_required_supabase_client():
    """Return a ready Supabase client or raise when unavailable."""
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured or currently unavailable.")
    return client

def is_supabase_enabled() -> bool:
    """Return True when Supabase env variables are configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)

def is_dns_resolution_error(exception: Exception) -> bool:
    """Return True when exception text indicates DNS hostname resolution failed."""
    error_text = str(exception).lower()
    patterns = [
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "failed to resolve",
        "getaddrinfo failed",
    ]
    return any(pattern in error_text for pattern in patterns)

@st.cache_resource
def get_supabase_client():
    """Create and cache Supabase client."""
    if st.session_state.get("supabase_sync_temporarily_disabled"):
        return None
    if not is_supabase_enabled():
        return None
    try:
        parsed = urlparse(SUPABASE_URL)
        if not parsed.scheme or not parsed.netloc:
            return None
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def sync_payload_to_supabase(table_name: str, payload: dict, on_conflict: str) -> None:
    """Upsert payload into Supabase and track sync errors in session state."""
    client = get_supabase_client()
    if client is None:
        return
    try:
        client.table(table_name).upsert(payload, on_conflict=on_conflict).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        if is_dns_resolution_error(exc):
            st.session_state["supabase_sync_temporarily_disabled"] = True
            st.session_state["last_supabase_sync_error"] = (
                "Supabase host could not be resolved. Data operations are paused "
                "for this session until connectivity is restored."
            )
            return
        st.session_state["last_supabase_sync_error"] = f"{table_name} sync failed: {exc}"

def sync_user_to_supabase(user_payload: dict) -> None:
    """Mirror user record to Supabase when configured."""
    sync_payload_to_supabase("users", user_payload, "user_id")

def sync_profile_to_supabase(profile_payload: dict) -> None:
    """Mirror profile record to Supabase when configured."""
    sync_payload_to_supabase("profile", profile_payload, "user_id")

def sync_daily_input_to_supabase(daily_payload: dict) -> None:
    """Mirror daily input record to Supabase when configured."""
    sync_payload_to_supabase("daily_inputs", daily_payload, "user_id,input_date")


def sync_local_history_to_supabase(user_id: str) -> int:
    """Read local `USER_RESPONSES_JSON_PATH` and upsert any rows that match `user_id`.

    Returns the number of rows attempted to sync.
    """
    if not user_id or not is_supabase_enabled():
        return 0

    if not os.path.exists(USER_RESPONSES_JSON_PATH):
        return 0

    synced = 0
    try:
        with open(USER_RESPONSES_JSON_PATH, "r", encoding="utf-8") as f:
            local_rows = json.load(f) or []
    except Exception:
        return 0

    for row in local_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("user_id") or "") != str(user_id):
            continue

        # Determine input_date and submitted_at
        submitted_at = row.get("timestamp") or row.get("submitted_at") or None
        input_date = None
        if isinstance(row.get("input_date"), str) and row.get("input_date"):
            input_date = row.get("input_date")
        elif isinstance(submitted_at, str) and len(submitted_at) >= 10:
            input_date = submitted_at[:10]

        if not input_date:
            # skip rows without a sensible date
            continue

        payload = {
            "user_id": user_id,
            "input_date": input_date,
            "submitted_at": submitted_at,
            "answers_json": row.get("user_inputs") or row.get("answers_json") or {},
            "prediction_json": row.get("predictions") or row.get("prediction_json") or {},
            "cluster": int(row.get("cluster") or 0),
        }

        try:
            sync_daily_input_to_supabase(payload)
            synced += 1
        except Exception:
            # Best-effort: continue to next row on errors
            continue

    return synced

def init_database() -> None:
    """Validate Supabase connectivity for required app tables."""
    try:
        client = get_required_supabase_client()
        client.table("users").select("user_id").limit(1).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"Supabase setup check failed: {exc}"

    # Authentication and profile data access
def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hash password with PBKDF2-HMAC-SHA256 and a per-user salt."""
    salt_value = salt or secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_value.encode("utf-8"),
        200000,
    )
    return hashed.hex(), salt_value

def verify_password(password: str, expected_hash: str, salt: str) -> bool:
    """Verify password against stored hash and salt."""
    calculated_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated_hash, expected_hash)

def create_user(first_name: str, last_name: str, email: str, password: str, phone_number: str = "", city: str = "", zip_code: str = "") -> tuple[bool, str]:
    """Create a user account in Supabase."""
    normalized_email = email.strip().lower()
    password_hash, password_salt = hash_password(password)
    user_id = str(uuid.uuid4())

    try:
        client = get_required_supabase_client()
        client.table("users").insert({"user_id": user_id, "first_name": first_name.strip(), "last_name": last_name.strip(), "email": normalized_email, "phone_number": phone_number.strip() or None, "city": city.strip() or None, "zip_code": zip_code.strip() or None, "password_hash": password_hash, "password_salt": password_salt, "is_verified": False}).execute()
        st.session_state["last_supabase_sync_error"] = None
        return True, user_id
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"users create failed: {exc}"
        error_text = str(exc).lower()
        if "row-level security" in error_text or "rls" in error_text:
            return False, "Signup is blocked by Supabase row-level security on the users table. Add an insert policy for anonymous signups or use a service role key."
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            return False, "A user with this email already exists."
        return False, "Unable to create account right now. Please try again."


def authenticate_user(email: str, password: str) -> tuple[bool, Optional[dict], str]:
    """Authenticate user by email and password."""
    normalized_email = email.strip().lower()
    try:
        client = get_required_supabase_client()
        response = (
            client.table("users")
            .select("user_id,first_name,last_name,email,phone_number,password_hash,password_salt")
            .eq("email", normalized_email)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        row = rows[0] if rows else None
    except Exception as exc:
        if is_dns_resolution_error(exc):
            return False, None, "Login service is unavailable because Supabase host resolution failed. Check your network connection."
        return False, None, f"Login service error: {exc}"

    if not row:
        return False, None, "No account found with this email."

    if not isinstance(row, dict):
        return False, None, "Login service returned unexpected data format."

    user_data = {
        "user_id": str(row.get("user_id") or ""),
        "first_name": str(row.get("first_name") or ""),
        "last_name": str(row.get("last_name") or ""),
        "email": str(row.get("email") or ""),
        "phone_number": str(row.get("phone_number") or ""),
    }
    expected_hash = str(row.get("password_hash") or "")
    password_salt = str(row.get("password_salt") or "")
    if not verify_password(password, expected_hash, password_salt):
        return False, None, "Invalid password."

    return True, user_data, ""


def upsert_profile(user_id: str, age: Optional[int], lifestyle_parameters: str, personal_details: str) -> bool:
    """Create or update profile details for a user."""
    try:
        client = get_required_supabase_client()
        client.table("profile").upsert(
            {
                "user_id": user_id,
                "age": age,
                "lifestyle_parameters": {"text": lifestyle_parameters.strip()} if lifestyle_parameters.strip() else {},
                "personal_details": {"text": personal_details.strip()} if personal_details.strip() else {},
            },
            on_conflict="user_id",
        ).execute()
        st.session_state["last_supabase_sync_error"] = None
        return True
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"profile upsert failed: {exc}"
        return False


def get_user_static_answers(user_id: str) -> dict:
    """Return saved baseline answers for a user, if present."""
    try:
        client = get_required_supabase_client()
        response = (
            client.table("users")
            .select("survey_age,survey_course,survey_gender,survey_cgpa,survey_relationship,survey_family_history,survey_semester,survey_residence_type")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        row = rows[0] if rows else None
    except Exception as exc:
        if _is_missing_users_static_columns_error(exc):
            return get_profile_static_answers(user_id)
        return {}

    if not isinstance(row, dict):
        return get_profile_static_answers(user_id)

    return {
        "Age": row.get("survey_age"),
        "Course": row.get("survey_course"),
        "Gender": row.get("survey_gender"),
        "CGPA": row.get("survey_cgpa"),
        "Relationship": row.get("survey_relationship"),
        "Family_History": row.get("survey_family_history"),
        "Semester": row.get("survey_semester"),
        "Residence_Type": row.get("survey_residence_type"),
    }


def _is_missing_users_static_columns_error(exception: Exception) -> bool:
    """Return True when Supabase reports missing users.survey_* columns."""
    text = str(exception).lower()
    return "pgrst204" in text or ("survey_age" in text and "schema cache" in text)


def _parse_json_like(value: Any) -> dict:
    """Parse dict-or-JSON-string payloads safely into a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _read_profile_personal_details(user_id: str) -> dict:
    """Read profile.personal_details for the user as a dictionary."""
    try:
        client = get_required_supabase_client()
        response = (
            client.table("profile")
            .select("personal_details")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        row = rows[0] if rows else None
    except Exception:
        return {}

    if not isinstance(row, dict):
        return {}
    return _parse_json_like(row.get("personal_details"))


def get_profile_static_answers(user_id: str) -> dict:
    """Read static survey answers stored inside profile.personal_details."""
    details = _read_profile_personal_details(user_id)
    static_blob = details.get("static_answers", {}) if isinstance(details, dict) else {}
    static_blob = _parse_json_like(static_blob)

    return {
        field_name: (static_blob.get(field_name) or "")
        for field_name in STATIC_USER_FIELDS
    }


def save_profile_static_answers(user_id: str, answers: dict, only_if_missing: bool = False) -> None:
    """Persist static survey answers in profile.personal_details.static_answers."""
    existing_details = _read_profile_personal_details(user_id)
    static_blob = _parse_json_like(existing_details.get("static_answers", {}))

    for field_name in STATIC_USER_FIELDS:
        incoming = answers.get(field_name)
        value = incoming.strip() if isinstance(incoming, str) else incoming
        if not value:
            continue
        current_value = str(static_blob.get(field_name) or "").strip()
        if only_if_missing and current_value:
            continue
        static_blob[field_name] = value

    if not static_blob:
        return

    existing_details["static_answers"] = static_blob
    try:
        client = get_required_supabase_client()
        client.table("profile").upsert(
            {
                "user_id": user_id,
                "personal_details": existing_details,
            },
            on_conflict="user_id",
        ).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"profile static answers update failed: {exc}"


def get_latest_daily_static_answers(user_id: str) -> dict:
    """Return static profile fields from the user's most recent saved daily survey."""
    try:
        client = get_required_supabase_client()
        response = (
            client.table("daily_inputs")
            .select("answers_json")
            .eq("user_id", user_id)
            .order("submitted_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        row = rows[0] if rows else None
    except Exception:
        return {}

    if not isinstance(row, dict):
        return {}

    answers_payload = row.get("answers_json")
    if isinstance(answers_payload, str):
        try:
            answers_payload = json.loads(answers_payload)
        except Exception:
            answers_payload = {}

    if not isinstance(answers_payload, dict):
        return {}

    static_answers = {}
    for field_name in STATIC_USER_FIELDS:
        value = (answers_payload.get(field_name) or "")
        if isinstance(value, str):
            value = value.strip()
        if value:
            static_answers[field_name] = value
    return static_answers


def get_merged_static_answers(user_id: str) -> dict:
    """Merge static answers from users columns, profile fallback, and latest daily input."""
    users_static = get_user_static_answers(user_id)
    profile_static = get_profile_static_answers(user_id)
    latest_daily_static = get_latest_daily_static_answers(user_id)

    merged: dict[str, Any] = {}
    for field_name in STATIC_USER_FIELDS:
        users_value = str(users_static.get(field_name) or "").strip()
        profile_value = str(profile_static.get(field_name) or "").strip()
        daily_value = str(latest_daily_static.get(field_name) or "").strip()
        merged[field_name] = users_value or profile_value or daily_value
    return merged

def save_user_static_answer_if_missing(user_id: str, field_name: str, field_value: str) -> None:
    """Persist first submitted baseline answer only once for the user."""
    column_name = STATIC_USER_FIELDS.get(field_name)
    value = str(field_value or "").strip()
    if not column_name or not value:
        return

    try:
        client = get_required_supabase_client()
        current_response = (
            client.table("users")
            .select(column_name)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = current_response.data or []
        row = rows[0] if rows else None
        current_value = row.get(column_name) if isinstance(row, dict) else ""
        if str(current_value).strip():
            return

        client.table("users").update({column_name: value}).eq("user_id", user_id).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        if _is_missing_users_static_columns_error(exc):
            save_profile_static_answers(user_id, {field_name: value}, only_if_missing=True)
            return
        st.session_state["last_supabase_sync_error"] = f"users static answer update failed: {exc}"


def save_user_static_answers(user_id: str, answers: dict) -> None:
    """Persist editable static survey answers for a logged-in user."""
    payload = {}
    for field_name, column_name in STATIC_USER_FIELDS.items():
        value = str(answers.get(field_name) or "").strip()
        if value:
            payload[column_name] = value

    if not payload:
        return

    try:
        client = get_required_supabase_client()
        client.table("users").update(payload).eq("user_id", user_id).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        if _is_missing_users_static_columns_error(exc):
            save_profile_static_answers(user_id, answers)
            return
        st.session_state["last_supabase_sync_error"] = f"users static answers update failed: {exc}"


def get_next_required_question(required_fields: list, answers: dict) -> Optional[str]:
    """Return the first required survey field that is still unanswered."""
    for field_name in required_fields:
        value = answers.get(field_name)
        if value is None:
            return field_name
        if isinstance(value, str) and not value.strip():
            return field_name
    return None


def count_answered_required_fields(required_fields: list, answers: dict) -> int:
    """Count answered required fields in the survey flow."""
    answered_count = 0
    for field_name in required_fields:
        value = answers.get(field_name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        answered_count += 1
    return answered_count


def save_user_daily_input_to_sql(user_id: str, answers: dict, prediction: dict, cluster: int) -> tuple[bool, str]:
    """Save one daily questionnaire response per user to Supabase."""
    today_value = date.today().isoformat()
    submitted_at = datetime.now().isoformat()

    try:
        client = get_required_supabase_client()
        client.table("daily_inputs").insert({"user_id": user_id, "input_date": today_value, "submitted_at": submitted_at, "answers_json": answers, "prediction_json": prediction, "cluster": cluster}).execute()
        st.session_state["last_data_save_error"] = None
        st.session_state["last_supabase_sync_error"] = None
        return True, ""
    except Exception as exc:
        error_text = str(exc).lower()
        if "duplicate" in error_text or "unique" in error_text:
            return False, "You have already submitted today's input. Please come back tomorrow."
        st.session_state["last_data_save_error"] = f"daily_inputs save failed: {exc}"
        st.session_state["last_supabase_sync_error"] = f"daily_inputs save failed: {exc}"
        return False, "Unable to save your daily input right now."


def build_demo_prediction_payload(day_index: int) -> dict:
    """Create a deterministic burnout prediction payload for demo history rows."""
    demo_steps = [
        ("High Burnout", {"low burnout": 0.08, "mid burnout": 0.17, "high burnout": 0.75}),
        ("High Burnout", {"low burnout": 0.10, "mid burnout": 0.20, "high burnout": 0.70}),
        ("High Burnout", {"low burnout": 0.12, "mid burnout": 0.24, "high burnout": 0.64}),
        ("Mid Burnout", {"low burnout": 0.18, "mid burnout": 0.48, "high burnout": 0.34}),
        ("Mid Burnout", {"low burnout": 0.24, "mid burnout": 0.50, "high burnout": 0.26}),
        ("Mid Burnout", {"low burnout": 0.30, "mid burnout": 0.47, "high burnout": 0.23}),
        ("Low Burnout", {"low burnout": 0.48, "mid burnout": 0.34, "high burnout": 0.18}),
        ("Low Burnout", {"low burnout": 0.56, "mid burnout": 0.28, "high burnout": 0.16}),
        ("Low Burnout", {"low burnout": 0.63, "mid burnout": 0.22, "high burnout": 0.15}),
        ("Low Burnout", {"low burnout": 0.70, "mid burnout": 0.18, "high burnout": 0.12}),
    ]
    predicted_class, probabilities = demo_steps[min(max(day_index, 0), len(demo_steps) - 1)]
    return {
        "random_forest": {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
        }
    }


def ensure_demo_history_for_test_account(user_id: str, current_email: str) -> bool:
    """Seed a one-time 10-day demo history for the test account."""
    if not user_id or (current_email or "").strip().lower() != TEST_DEMO_EMAIL:
        return False

    session_flag = f"demo_history_seeded:{user_id}"
    if st.session_state.get(session_flag):
        return False

    try:
        client = get_required_supabase_client()
        response = (
            client.table("daily_inputs")
            .select("input_date")
            .eq("user_id", user_id)
            .order("input_date", desc=False)
            .execute()
        )
        existing_dates = set()
        for row in response.data or []:
            if isinstance(row, dict):
                input_date_value = row.get("input_date")
                if input_date_value:
                    existing_dates.add(str(input_date_value))

        rows_to_insert = []
        for day_index, days_back in enumerate(range(9, -1, -1)):
            target_date = date.today() - timedelta(days=days_back)
            target_date_value = target_date.isoformat()
            if target_date_value in existing_dates:
                continue
            rows_to_insert.append(
                {
                    "user_id": user_id,
                    "input_date": target_date_value,
                    "submitted_at": f"{target_date_value}T09:00:00",
                    "answers_json": {},
                    "prediction_json": build_demo_prediction_payload(day_index),
                    "cluster": 0,
                }
            )

        if rows_to_insert:
            client.table("daily_inputs").insert(rows_to_insert).execute()

        st.session_state[session_flag] = True
        return bool(rows_to_insert)
    except Exception as exc:
        st.session_state["last_data_save_error"] = f"demo history seed failed: {exc}"
        return False

def has_user_submitted_today(user_id: str) -> bool:
    """Check whether user already submitted daily survey today."""
    today_value = date.today().isoformat()
    try:
        client = get_required_supabase_client()
        response = (
            client.table("daily_inputs")
            .select("entry_id")
            .eq("user_id", user_id)
            .eq("input_date", today_value)
            .limit(1)
            .execute()
        )
        return bool(response.data)
    except Exception:
        return False


def upsert_daily_feedback(user_id: str, input_date: str, recommendation_followed: Optional[bool], recommendation_helpful: Optional[bool], feedback_rating: Optional[int], app_feedback: str) -> tuple[bool, str]:
    """Create or update per-day recommendation/app feedback for a user."""
    try:
        client = get_required_supabase_client()
        client.table("daily_feedback").upsert(
            {
                "user_id": user_id,
                "input_date": input_date,
                "recommendation_followed": None if recommendation_followed is None else int(recommendation_followed),
                "recommendation_helpful": None if recommendation_helpful is None else int(recommendation_helpful),
                "feedback_rating": feedback_rating,
                "app_feedback": app_feedback.strip() or None,
            },
            on_conflict="user_id,input_date",
        ).execute()
        st.session_state["last_supabase_sync_error"] = None
        return True, ""
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"daily_feedback save failed: {exc}"
        return False, "Unable to save feedback right now."


def get_user_daily_history(user_id: str) -> pd.DataFrame:
    """Fetch a user's day-by-day survey history with optional feedback."""
    client = None
    try:
        client = get_required_supabase_client()
        daily_response = (
            client.table("daily_inputs")
            .select("entry_id,user_id,input_date,submitted_at,prediction_json,cluster")
            .eq("user_id", user_id)
            .order("input_date", desc=False)
            .execute()
        )
    except Exception:
        # If Supabase is unavailable or query failed, we'll fall back to local JSON
        # history saved by `save_user_response_to_json` (if present).
        daily_response = None

    history_df = pd.DataFrame(daily_response.data or []) if daily_response is not None else pd.DataFrame()

    # If Supabase returned nothing, try local JSON fallback for this user.
    if history_df.empty:
        try:
            if os.path.exists(USER_RESPONSES_JSON_PATH):
                with open(USER_RESPONSES_JSON_PATH, "r", encoding="utf-8") as f:
                    local_rows = json.load(f) or []
                matched_rows = []
                for row in local_rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("user_id") or "") != str(user_id or ""):
                        continue
                    # Map local JSON shape to the same fields we expect from Supabase
                    ts = row.get("timestamp") or row.get("submitted_at") or None
                    try:
                        input_date = ts[:10] if isinstance(ts, str) and len(ts) >= 10 else None
                    except Exception:
                        input_date = None
                    matched_rows.append(
                        {
                            "entry_id": row.get("entry_id") or str(uuid.uuid4()),
                            "user_id": row.get("user_id"),
                            "input_date": input_date,
                            "submitted_at": ts,
                            "prediction_json": row.get("predictions") or row.get("prediction_json") or {},
                            "cluster": row.get("cluster") or 0,
                        }
                    )
                if matched_rows:
                    history_df = pd.DataFrame(matched_rows)
        except Exception:
            history_df = pd.DataFrame()

    if history_df.empty:
        return history_df

    feedback_df = pd.DataFrame()
    if client is not None and daily_response is not None:
        try:
            feedback_response = (
                client.table("daily_feedback")
                .select("user_id,input_date,recommendation_followed,recommendation_helpful,feedback_rating,app_feedback")
                .eq("user_id", user_id)
                .execute()
            )
            feedback_df = pd.DataFrame(feedback_response.data or [])
        except Exception:
            feedback_df = pd.DataFrame()

    if not feedback_df.empty:
        history_df = history_df.merge(
            feedback_df,
            on=["user_id", "input_date"],
            how="left",
        )
    else:
        history_df["recommendation_followed"] = None
        history_df["recommendation_helpful"] = None
        history_df["feedback_rating"] = None
        history_df["app_feedback"] = None

    parsed_predictions = history_df["prediction_json"].apply(parse_prediction_json)
    trend_metrics = parsed_predictions.apply(derive_trend_metrics_from_prediction)
    history_df["predicted_class"] = parsed_predictions.apply(derive_prediction_class_from_prediction)
    history_df["stress_level"] = trend_metrics.apply(lambda t: t.get("stress_level"))
    history_df["anxiety_score"] = trend_metrics.apply(lambda t: t.get("anxiety_score"))
    history_df["depression_score"] = trend_metrics.apply(lambda t: t.get("depression_score"))
    history_df["mental_health_pct"] = trend_metrics.apply(lambda t: t.get("mental_health_pct"))
    return history_df


def fetch_daily_inputs_from_supabase(user_id: str) -> pd.DataFrame:
    """Attempt to fetch daily_inputs rows for a user directly from Supabase.

    This is a non-fallback helper that returns an empty DataFrame when Supabase
    is not available or the query fails.
    """
    try:
        client = get_required_supabase_client()
        response = (
            client.table("daily_inputs")
            .select("entry_id,user_id,input_date,submitted_at,prediction_json,cluster")
            .eq("user_id", user_id)
            .order("input_date", desc=False)
            .execute()
        )
    except Exception:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(response.data or [])
        return df
    except Exception:
        return pd.DataFrame()


def parse_prediction_json(prediction_json: Any) -> dict:
    """Safely parse prediction JSON payload."""
    try:
        if not prediction_json:
            return {}
        if isinstance(prediction_json, dict):
            return prediction_json
        parsed = json.loads(prediction_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def get_admin_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch users and daily input records for admin view."""
    try:
        client = get_required_supabase_client()
        users_response = (
            client.table("users")
            .select("user_id,first_name,last_name,email,phone_number,city,zip_code,created_at")
            .order("created_at", desc=True)
            .execute()
        )
        daily_response = (
            client.table("daily_inputs")
            .select("entry_id,user_id,input_date,submitted_at,prediction_json,cluster")
            .order("submitted_at", desc=True)
            .execute()
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    users_df = pd.DataFrame(users_response.data or [])
    daily_df = pd.DataFrame(daily_response.data or [])

    if users_df.empty:
        users_df = pd.DataFrame(columns=[
            "user_id", "first_name", "last_name", "email", "phone_number", "city", "zip_code", "created_at", "total_entries", "last_input_date"
        ])
    if daily_df.empty:
        daily_df = pd.DataFrame(columns=["entry_id", "user_id", "input_date", "submitted_at", "prediction_json", "cluster"])
    else:
        aggregates = (
            daily_df.groupby("user_id", as_index=False)
            .agg(total_entries=("entry_id", "count"), last_input_date=("input_date", "max"))
        )
        users_df = users_df.merge(aggregates, on="user_id", how="left")

    users_df["total_entries"] = users_df["total_entries"].fillna(0).astype(int)

    if not daily_df.empty:
        parsed_predictions = daily_df["prediction_json"].apply(parse_prediction_json)
        trend_metrics = parsed_predictions.apply(derive_trend_metrics_from_prediction)
        daily_df["stress_level"] = trend_metrics.apply(lambda t: t.get("stress_level"))
        daily_df["anxiety_score"] = trend_metrics.apply(lambda t: t.get("anxiety_score"))
        daily_df["depression_score"] = trend_metrics.apply(lambda t: t.get("depression_score"))

    return users_df, daily_df

# UI renderers and app navigation
def render_admin_page() -> None:
    """Render a simple admin page for users and daily trends."""
    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    st.markdown("<div class='middle-section'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Admin View</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Users and Daily History Trends</h2>", unsafe_allow_html=True)

    users_df, daily_df = get_admin_data()

    total_users = int(len(users_df))
    total_entries = int(len(daily_df))
    today_entries = int((daily_df["input_date"] == date.today().isoformat()).sum()) if not daily_df.empty else 0

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("Total Users", total_users)
    with metric_col_2:
        st.metric("Total Daily Entries", total_entries)
    with metric_col_3:
        st.metric("Today's Entries", today_entries)

    st.markdown("#### 👥 Users")
    if users_df.empty:
        st.info("No users found yet.")
    else:
        users_display = users_df[[
            "first_name", "last_name", "email", "phone_number", "city", "zip_code", "total_entries", "last_input_date"
        ]].copy()
        users_display = users_display.rename(columns={"first_name": "First Name", "last_name": "Last Name", "email": "Email", "phone_number": "Phone", "city": "City", "zip_code": "ZIP", "total_entries": "Entries", "last_input_date": "Last Input Date"})
        st.dataframe(users_display, hide_index=True)

    st.markdown("#### 📈 Daily Trend")
    if daily_df.empty:
        st.info("No daily input history found yet.")
    else:
        trend_df = (
            daily_df.groupby("input_date", as_index=False)
            .agg(submissions=("entry_id", "count"), avg_stress=("stress_level", "mean"))
            .sort_values("input_date")
        )

        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Bar(
                x=trend_df["input_date"],
                y=trend_df["submissions"],
                name="Submissions",
                marker_color="#5B7FEA",
                yaxis="y",
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=trend_df["input_date"],
                y=trend_df["avg_stress"],
                mode="lines+markers",
                name="Avg Stress",
                marker_color="#9B7FB5",
                yaxis="y2",
            )
        )
        trend_fig.update_layout(
            xaxis_title="Date",
            yaxis=dict(title="Submissions"),
            yaxis2=dict(title="Avg Stress", overlaying="y", side="right", range=[0, 5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(trend_fig, width="stretch")

        st.markdown("#### 🧾 User Daily History")
        user_option_map = {
            f"{row['first_name']} {row['last_name']} ({row['email']})": row["user_id"]
            for _, row in users_df.iterrows()
        }
        selected_user_label = st.selectbox("Select user", options=list(user_option_map.keys()))
        selected_user_id = user_option_map[selected_user_label]

        user_history_df = daily_df[daily_df["user_id"] == selected_user_id].copy()
        user_history_df = user_history_df.sort_values("submitted_at")

        if user_history_df.empty:
            st.info("This user has no daily entries yet.")
        else:
            user_stress_fig = go.Figure()
            user_stress_fig.add_trace(
                go.Scatter(
                    x=user_history_df["input_date"],
                    y=user_history_df["stress_level"],
                    mode="lines+markers",
                    name="Stress Level",
                    marker_color="#3F2456",
                )
            )
            user_stress_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Stress Level",
                yaxis=dict(range=[0, 5]),
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(user_stress_fig, width="stretch")

            history_display = user_history_df[[
                "input_date", "submitted_at", "stress_level", "anxiety_score", "depression_score", "cluster"
            ]].copy()
            history_display = history_display.rename(
                columns={
                    "input_date": "Date",
                    "submitted_at": "Submitted At",
                    "stress_level": "Stress",
                    "anxiety_score": "Anxiety %",
                    "depression_score": "Depression %",
                    "cluster": "Cluster",
                }
            )
            st.dataframe(history_display, hide_index=True)

    if st.button("← Back to AuraCheck", key="admin_back_to_main", width="stretch"):
        st.session_state["auth_page"] = "main"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def save_user_response_to_json(answers: dict, prediction: dict, cluster: int, user_id: Optional[str] = None) -> None:
    """Save user response to JSON file. Include `user_id` when available so local history
    can be used as a fallback when Supabase is unavailable or prior anonymous runs were
    performed before login.
    """
    try:
        response_data = {
            "entry_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_inputs": answers,
            "predictions": prediction,
            "cluster": cluster,
        }

        all_responses = []

        if os.path.exists(USER_RESPONSES_JSON_PATH):
            with open(USER_RESPONSES_JSON_PATH, "r", encoding="utf-8") as f:
                try:
                    all_responses = json.load(f)
                except Exception:
                    all_responses = []

        all_responses.append(response_data)

        with open(USER_RESPONSES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_responses, f, indent=2)

    except Exception:
        # JSON export is optional and should not block the UI flow.
        pass


def initialize_state() -> None:
    """Initialize session state."""
    defaults = {
        "last_answers": {},
        "last_prediction": None,
        "last_cluster": None,
        "show_results": False,
        "result_model_choice": "Random Forest",
        "auth_page": "main",
        "current_user_id": None,
        "current_user_email": None,
        "current_user_name": None,
        "last_data_save_error": None,
        "last_supabase_sync_error": None,
        "supabase_sync_temporarily_disabled": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_survey_state() -> None:
    """Reset questionnaire and latest result state."""
    st.session_state["last_answers"] = {}
    st.session_state["last_prediction"] = None
    st.session_state["last_cluster"] = None
    st.session_state["show_results"] = False


def get_selected_model_output(prediction: Optional[dict], selected_model: str) -> tuple[str, str, float, dict]:
    """Return random forest display label, score label, score value, and details."""
    rf_output = prediction.get("random_forest", {}) if isinstance(prediction, dict) else {}
    score = float(rf_output.get("accuracy", 0.0))
    return (
        str(rf_output.get("predicted_class", "N/A")),
        "Accuracy %",
        score,
        rf_output,
    )


def compact_model_outputs(prediction_payload: dict) -> dict:
    """Keep only the random forest output for storage and rendering."""
    if not isinstance(prediction_payload, dict):
        return {}
    return {
        "random_forest": prediction_payload.get("random_forest", {}),
    }


def derive_trend_metrics_from_prediction(prediction_payload: dict) -> dict:
    """Derive trend metrics from prediction payload (supports old and new schema)."""
    if not isinstance(prediction_payload, dict):
        return {
            "stress_level": None,
            "anxiety_score": None,
            "depression_score": None,
            "mental_health_pct": None,
        }

    # Backward compatibility with prior payload format.
    if any(k in prediction_payload for k in ["stress_level", "anxiety_score", "depression_score", "mental_health_pct"]):
        return {
            "stress_level": prediction_payload.get("stress_level"),
            "anxiety_score": prediction_payload.get("anxiety_score"),
            "depression_score": prediction_payload.get("depression_score"),
            "mental_health_pct": prediction_payload.get("mental_health_pct"),
        }

    rf_output = prediction_payload.get("random_forest", {}) if isinstance(prediction_payload.get("random_forest", {}), dict) else {}
    probs = rf_output.get("probabilities", {}) if isinstance(rf_output.get("probabilities", {}), dict) else {}
    low_prob = float(probs.get("low burnout", 0.0))
    mid_prob = float(probs.get("mid burnout", 0.0))
    high_prob = float(probs.get("high burnout", 0.0))

    if (low_prob + mid_prob + high_prob) > 0:
        stress_level = max(1.0, min(5.0, (low_prob * 2.0) + (mid_prob * 3.0) + (high_prob * 4.5)))
        mental_health_pct = max(0.0, min(100.0, (low_prob * 90.0) + (mid_prob * 55.0) + (high_prob * 20.0)))
        anxiety_score = max(0.0, min(100.0, (high_prob * 85.0) + (mid_prob * 55.0) + (low_prob * 25.0)))
        depression_score = max(0.0, min(100.0, (high_prob * 80.0) + (mid_prob * 50.0) + (low_prob * 20.0)))
        return {
            "stress_level": float(round(stress_level, 2)),
            "anxiety_score": float(round(anxiety_score, 2)),
            "depression_score": float(round(depression_score, 2)),
            "mental_health_pct": float(round(mental_health_pct, 2)),
        }

    # Fallback if probability dictionary is unavailable.
    cls = str(rf_output.get("predicted_class", "")).strip().lower()
    fallback_map = {
        "low burnout": {"stress_level": 2.0, "anxiety_score": 30.0, "depression_score": 28.0, "mental_health_pct": 78.0},
        "mid burnout": {"stress_level": 3.0, "anxiety_score": 55.0, "depression_score": 52.0, "mental_health_pct": 52.0},
        "high burnout": {"stress_level": 4.2, "anxiety_score": 78.0, "depression_score": 76.0, "mental_health_pct": 24.0},
    }
    return fallback_map.get(
        cls,
        {"stress_level": None, "anxiety_score": None, "depression_score": None, "mental_health_pct": None},
    )
def derive_prediction_class_from_prediction(prediction_payload: dict) -> str:
    """Return a human-readable predicted burnout class from the stored model outputs."""
    if not isinstance(prediction_payload, dict):
        return "Unknown"

    rf_output = prediction_payload.get("random_forest", {}) if isinstance(prediction_payload.get("random_forest", {}), dict) else {}
    class_label = str(rf_output.get("predicted_class", "")).strip()
    if class_label:
        return class_label

    baseline_output = prediction_payload.get("baseline_multinomial", {}) if isinstance(prediction_payload.get("baseline_multinomial", {}), dict) else {}
    class_label = str(baseline_output.get("predicted_class", "")).strip()
    if class_label:
        return class_label

    kmeans_output = prediction_payload.get("unsupervised_kmeans", {}) if isinstance(prediction_payload.get("unsupervised_kmeans", {}), dict) else {}
    class_label = str(kmeans_output.get("mapped_burnout_class", "")).strip()
    return class_label or "Unknown"


def normalize_prediction_class_label(class_label: str) -> str:
    """Normalize class names so the history view can present a consistent progression scale."""
    text = (class_label or "").strip().lower()
    if not text:
        return "Unknown"
    if "very low" in text or "q1" in text:
        return "Very Low Burnout"
    if "low" in text or "q2" in text:
        return "Low Burnout"
    if "mid" in text or "moderate" in text or "q3" in text:
        return "Mid Burnout"
    if "high" in text or "q4" in text:
        return "High Burnout"
    return class_label


def prediction_class_to_score(class_label: str) -> float:
    """Map a burnout class label to an ordinal score for charting."""
    normalized = normalize_prediction_class_label(class_label).lower()
    if normalized == "very low burnout":
        return 0.0
    if normalized == "low burnout":
        return 1.0
    if normalized == "mid burnout":
        return 2.0
    if normalized == "high burnout":
        return 3.0
    return 1.5


def render_user_progress_section(user_id: str) -> None:
    """Render per-user prediction-class history and feedback form."""
    history_df = get_user_daily_history(user_id)

    st.markdown("#### 📈 Daily Progress (Status vs Date)")
    if history_df.empty:
        st.info("No saved daily history yet. Complete today's survey to start tracking burnout class changes.")
        return

    history_df = history_df.sort_values("input_date").copy()
    history_df["predicted_class_display"] = history_df["predicted_class"].apply(normalize_prediction_class_label)
    history_df["predicted_class_score"] = history_df["predicted_class_display"].apply(prediction_class_to_score)

    class_colors = {
        "Very Low Burnout": "#2E8B57",
        "Low Burnout": "#C39A3A",
        "Mid Burnout": "#A67C00",
        "High Burnout": "#A04857",
        "Unknown": "#7A6B8F",
    }

    class_fig = go.Figure()
    class_fig.add_trace(
        go.Scatter(
            x=history_df["input_date"],
            y=history_df["predicted_class_score"],
            mode="lines+markers+text",
            text=history_df["predicted_class_display"],
            textposition="top center",
            marker=dict(
                size=12,
                color=[class_colors.get(cls, "#7A6B8F") for cls in history_df["predicted_class_display"]],
                line=dict(color="#FFFFFF", width=2),
            ),
            line=dict(color="#355DCB", width=3),
            name="Predicted class",
        )
    )
    class_fig.update_layout(
        xaxis_title="Date",
        yaxis=dict(
            title="Burnout Status",
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["Very Low", "Low", "Mid", "High"],
            range=[-0.3, 3.3],
        ),
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    st.plotly_chart(class_fig, use_container_width=True)


def is_admin_user() -> bool:
    """Return True only for the allowed admin email."""
    current_email = (st.session_state.get("current_user_email") or "").strip().lower()
    return current_email == ADMIN_EMAIL


def all_required_answered(answers: dict, required_fields: list) -> bool:
    """Check that all required fields are answered with valid non-skipped values."""
    for field in required_fields:
        value = answers.get(field)
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or normalized.lower() == "skipped":
                return False
    return True


def render_auth_page(auth_page: str) -> None:
    """Render dedicated authentication pages."""
    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    st.markdown("<div class='middle-section'>", unsafe_allow_html=True)
    st.image("Dataset/logo.jpg", width=90)

    if auth_page == "signup":
        st.markdown("<h1 style='text-align: center;'>Join AuraCheck</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Create your account</h2>", unsafe_allow_html=True)

        with st.form("signup_form"):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            signup_email = st.text_input("Email Address")
            phone_number = st.text_input("Phone Number (Optional)")
            city = st.text_input("City (Optional)")
            zip_code = st.text_input("ZIP (Optional)")
            signup_password = st.text_input("Password", type="password")
            signup_confirm_password = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Sign Up", width="stretch")

        if signup_submit:
            if not first_name.strip() or not last_name.strip() or not signup_email.strip():
                st.warning("⚠️ Please fill in first name, last name, and email address.")
            elif "@" not in signup_email or "." not in signup_email:
                st.warning("⚠️ Please enter a valid email address.")
            elif len(signup_password) < 8:
                st.warning("⚠️ Password must be at least 8 characters.")
            elif signup_password != signup_confirm_password:
                st.warning("⚠️ Password and confirm password do not match.")
            else:
                created, message = create_user(
                    first_name=first_name,
                    last_name=last_name,
                    email=signup_email,
                    password=signup_password,
                    phone_number=phone_number,
                    city=city,
                    zip_code=zip_code,
                )
                if created:
                    st.success(f"✅ Verification link has been sent to {signup_email.strip().lower()}.")
                else:
                    st.warning(f"⚠️ {message}")

        if st.button("Already have an account? Log In", key="goto_login", width="stretch"):
            st.session_state["auth_page"] = "login"
            st.rerun()

    if auth_page == "login":
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Access your AuraCheck account</h2>", unsafe_allow_html=True)

        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Log In", width="stretch")

        if login_submit:
            if not login_email.strip() or not login_password.strip():
                st.warning("⚠️ Please enter your email and password.")
            else:
                is_valid, user_data, auth_message = authenticate_user(login_email, login_password)
                if is_valid and user_data:
                    st.session_state["current_user_id"] = user_data["user_id"]
                    st.session_state["current_user_email"] = user_data["email"]
                    st.session_state["current_user_name"] = f"{user_data['first_name']} {user_data['last_name']}"
                    st.session_state["current_user_phone_number"] = user_data.get("phone_number", "")
                    # Attempt to sync any local JSON history entries for this user to Supabase.
                    try:
                        synced_count = sync_local_history_to_supabase(user_data["user_id"])
                        if synced_count:
                            st.info(f"✅ Synced {synced_count} local history entr{'y' if synced_count==1 else 'ies'} to your account.")
                    except Exception:
                        # Best-effort; do not block login on sync failures.
                        pass
                    st.session_state["auth_page"] = "profile"
                    st.success("✅ Login successful.")
                    st.rerun()
                else:
                    st.warning(f"⚠️ {auth_message}")

        st.markdown("<h4 style='text-align: center;'>Forgot Password?</h4>", unsafe_allow_html=True)
        forgot_email = st.text_input("Email for password reset", key="forgot_email")
        if st.button("Send Reset Link", key="forgot_password_btn", width="stretch"):
            if not forgot_email.strip():
                st.warning("⚠️ Please enter your email address.")
            elif "@" not in forgot_email or "." not in forgot_email:
                st.warning("⚠️ Please enter a valid email address.")
            else:
                st.success(f"✅ Reset link has been sent to {forgot_email.strip()}.")

        if st.button("Need an account? Sign Up", key="goto_signup", width="stretch"):
            st.session_state["auth_page"] = "signup"
            st.rerun()

    if st.button("← Back to AuraCheck", key="back_to_main", width="stretch"):
        st.session_state["auth_page"] = "main"
        st.rerun()

    if auth_page == "profile":
        st.markdown("<h1 style='text-align: center;'>Profile</h1>", unsafe_allow_html=True)
        current_name = st.session_state.get("current_user_name") or "User"
        current_email = st.session_state.get("current_user_email") or ""
        current_phone = st.session_state.get("current_user_phone_number") or ""
        current_user_id = st.session_state.get("current_user_id")
        st.markdown(f"<h2 style='text-align: center;'>Welcome, {current_name}</h2>", unsafe_allow_html=True)
        if current_email:
            st.markdown(f"<p style='text-align: center;'>Logged in as {current_email}</p>", unsafe_allow_html=True)
        if current_phone:
            st.markdown(f"<p style='text-align: center;'>Phone: {current_phone}</p>", unsafe_allow_html=True)

        name_parts = current_name.split(" ", 1)
        first_name_value = name_parts[0] if name_parts else ""
        last_name_value = name_parts[1] if len(name_parts) > 1 else ""

        st.markdown("#### Account Details")
        account_col_1, account_col_2 = st.columns(2)
        with account_col_1:
            st.text_input("First Name", value=first_name_value, disabled=True)
        with account_col_2:
            st.text_input("Last Name", value=last_name_value, disabled=True)
        detail_col_1, detail_col_2 = st.columns(2)
        with detail_col_1:
            st.text_input("Email", value=current_email, disabled=True)
        with detail_col_2:
            st.text_input("Phone Number", value=current_phone, disabled=True)

        if current_user_id and ensure_demo_history_for_test_account(current_user_id, current_email):
            st.info("Seeded 10 days of demo history for the test account.")
            st.rerun()

        saved_static_answers = get_merged_static_answers(current_user_id) if current_user_id else {}
        latest_daily_static_answers = get_latest_daily_static_answers(current_user_id) if current_user_id else {}

        # Auto-fill profile details from latest saved survey answers when user static columns are blank.
        if current_user_id:
            merged_static_answers = dict(saved_static_answers)
            for field_name in STATIC_USER_FIELDS:
                saved_value = str(merged_static_answers.get(field_name) or "").strip()
                latest_value = str(latest_daily_static_answers.get(field_name) or "").strip()
                if not saved_value and latest_value:
                    merged_static_answers[field_name] = latest_value
            if merged_static_answers != saved_static_answers:
                save_user_static_answers(current_user_id, merged_static_answers)
            saved_static_answers = merged_static_answers

        static_profile_defaults = {
            "Age": str(saved_static_answers.get("Age") or "").strip(),
            "Course": str(saved_static_answers.get("Course") or "").strip(),
            "Gender": str(saved_static_answers.get("Gender") or "").strip(),
            "CGPA": str(saved_static_answers.get("CGPA") or "").strip(),
            "Relationship": str(saved_static_answers.get("Relationship") or "").strip(),
            "Family_History": str(saved_static_answers.get("Family_History") or "").strip(),
            "Semester": str(saved_static_answers.get("Semester") or "").strip(),
            "Residence_Type": str(saved_static_answers.get("Residence_Type") or "").strip(),
        }

        st.markdown("#### Personal Details")
        st.caption("Your answers from survey assessments are saved here. Edit and save them whenever they change.")
        
        has_any_saved = any(static_profile_defaults.values())
        if not has_any_saved:
            st.info("📝 Complete your first survey to populate these details automatically.")
        
        with st.form("profile_static_answers_form"):
            profile_col_1, profile_col_2 = st.columns(2)
            
            with profile_col_1:
                st.markdown("**Your Saved Answers**")
                age_value = st.number_input(
                    "Age",
                    min_value=10,
                    max_value=100,
                    step=1,
                    value=_format_age_value(static_profile_defaults["Age"]),
                    format="%d"
                )
                gender_value = st.text_input(
                    "Gender",
                    value=static_profile_defaults["Gender"],
                    placeholder="e.g., Female"
                )
                relationship_value = st.text_input(
                    "Relationship",
                    value=static_profile_defaults["Relationship"],
                    placeholder="e.g., In a relationship"
                )
                semester_value = st.text_input(
                    "Semester",
                    value=static_profile_defaults["Semester"],
                    placeholder="e.g., 18"
                )
            
            with profile_col_2:
                st.markdown("**Your Saved Answers**")
                course_value = st.text_input(
                    "Course",
                    value=static_profile_defaults["Course"],
                    placeholder="e.g., CS"
                )
                cgpa_value = st.number_input(
                    "CGPA",
                    min_value=0.0,
                    max_value=4.0,
                    step=0.01,
                    value=_format_cgpa_value(static_profile_defaults["CGPA"]),
                    format="%.2f"
                )
                family_history_value = st.text_input(
                    "Family History",
                    value=static_profile_defaults["Family_History"],
                    placeholder="e.g., No"
                )
                residence_value = st.text_input(
                    "Residence Type",
                    value=static_profile_defaults["Residence_Type"],
                    placeholder="e.g., Dorm"
                )
            
            profile_save = st.form_submit_button("💾 Save profile details", width="stretch")

        if profile_save and current_user_id:
            save_user_static_answers(
                current_user_id,
                {
                    "Age": str(int(age_value)),
                    "Course": course_value,
                    "Gender": gender_value,
                    "CGPA": f"{float(cgpa_value):.2f}".rstrip("0").rstrip("."),
                    "Relationship": relationship_value,
                    "Family_History": family_history_value,
                    "Semester": semester_value,
                    "Residence_Type": residence_value,
                },
            )
            st.success("✅ Profile details saved.")
            st.rerun()

        st.markdown("#### Daily History")
        st.caption("Your saved survey submissions and burnout-class trend are shown below.")
        if current_user_id:
            # Allow explicit refresh from Supabase for immediate verification
            refresh_col, spacer = st.columns([1, 3])
            with refresh_col:
                if st.button("🔁 Refresh history from Supabase", key="refresh_history_btn"):
                    sup_df = fetch_daily_inputs_from_supabase(current_user_id)
                    if not sup_df.empty:
                        st.success(f"Fetched {len(sup_df)} rows from Supabase.")
                    else:
                        st.info("No rows returned from Supabase or service unavailable.")
                    # Rerun so the profile page picks up any newly synced rows
                    st.rerun()

            render_user_progress_section(current_user_id)

        if st.session_state.get("last_data_save_error"):
            st.warning(f"Data save issue: {st.session_state.get('last_data_save_error')}")
        if st.session_state.get("last_supabase_sync_error"):
            st.warning(f"Supabase sync issue: {st.session_state.get('last_supabase_sync_error')}")

        if st.button("Continue to AuraCheck", key="profile_to_main", width="stretch"):
            st.session_state["auth_page"] = "main"
            st.rerun()

        if st.button("Log Out", key="logout_btn", width="stretch"):
            st.session_state["current_user_id"] = None
            st.session_state["current_user_email"] = None
            st.session_state["current_user_name"] = None
            st.session_state["current_user_phone_number"] = None
            st.session_state["auth_page"] = "main"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main Streamlit Application."""

    # Initialize required runtime state before rendering any page sections.
    initialize_state()
    init_database()

    if st.session_state.get("auth_page") == "admin":
        if is_admin_user():
            render_admin_page()
            return
        st.session_state["auth_page"] = "main"
        st.warning("⚠️ Admin view is restricted to authorized admin only.")

    if st.session_state.get("auth_page") in {"signup", "login", "profile"}:
        render_auth_page(str(st.session_state.get("auth_page") or "main"))
        return

    if st.session_state.get("auth_page") == "model_analysis":
        render_model_analysis_page()
        return
    
    # --- CONTENT PLACEHOLDER - Shiny purple background ---
    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    
    # --- CREATE 3-COLUMN LAYOUT ---
    left_col, middle_col, right_col = st.columns([1.15, 2.25, 0.95], gap="large")
    
    # ========== LEFT COLUMN: Header & Positive Thoughts ==========
    with left_col:
        logo_pad_top, logo_mid, logo_pad_bottom = st.columns([1.2, 2, 0.8])
        with logo_mid:
            st.image("Dataset/logo.jpg", width=90)
        st.markdown("<h1 style='text-align: center;'>AuraCheck</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Your Quick Wellbeing Check-in</h2>", unsafe_allow_html=True)
        
        # Good Thoughts Section
        st.markdown("<div class='good-thoughts-header'>✨ Thought for the Day: </div>", unsafe_allow_html=True)
        
        thoughts_js = json.dumps(POSITIVE_THOUGHTS)
        components.html(
            f"""
            <div style="display:flex; justify-content:center; width:100%; margin-top: 6px;">
                <div id="thought-card" style="
                    width: 92%;
                    min-height: 140px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    color:#4A2F66;
                    font-size:30px;
                    font-weight:700;
                    line-height:1.45;
                    padding:18px 20px;
                    border-radius:14px;
                    border:2px solid rgba(155,127,181,0.35);
                    background: rgba(255,255,255,0.32);
                    box-sizing: border-box;
                    transition: opacity 500ms ease-in-out;
                "></div>
            </div>
            <script>
                const thoughts = {thoughts_js};
                const thoughtCard = document.getElementById('thought-card');
                let lastIndex = -1;

                function nextThought() {{
                    if (!thoughtCard || thoughts.length === 0) return;
                    thoughtCard.style.opacity = 0;
                    setTimeout(() => {{
                        let index = Math.floor(Math.random() * thoughts.length);
                        if (thoughts.length > 1) {{
                            while (index === lastIndex) {{
                                index = Math.floor(Math.random() * thoughts.length);
                            }}
                        }}
                        lastIndex = index;
                        thoughtCard.textContent = thoughts[index];
                        thoughtCard.style.transition = 'opacity 500ms ease-in-out';
                        thoughtCard.style.opacity = 1;
                    }}, 450);
                }}

                nextThought();
                setInterval(nextThought, 3200);
            </script>
            """,
            height=190,
        )
    
    # ========== MIDDLE COLUMN: Questions & Analysis ==========
    with middle_col:
        st.markdown("<div class='middle-section'>", unsafe_allow_html=True)
        st.markdown("<div class='middle-panel'>", unsafe_allow_html=True)
        
        required_fields = REQUIRED_FIELDS
        
        answers = dict(st.session_state.get("last_answers", {}))
        current_user_id = st.session_state.get("current_user_id")

        # For logged-in users, preload baseline answers and skip re-asking them.
        if current_user_id:
            saved_static_answers = get_merged_static_answers(current_user_id)
            for field_name in STATIC_USER_FIELDS:
                saved_value = (saved_static_answers.get(field_name) or "").strip()
                if saved_value and not answers.get(field_name):
                    answers[field_name] = saved_value
            if answers != st.session_state.get("last_answers", {}):
                st.session_state["last_answers"] = answers

        current_question_idx = count_answered_required_fields(required_fields, answers)
        next_required_field = get_next_required_question(required_fields, answers)
        already_submitted_today = bool(current_user_id and has_user_submitted_today(current_user_id))

        if already_submitted_today:
            st.info("✅ You already submitted today's survey. Come back tomorrow for your next check-in.")
        
        # Progress Section
        if current_question_idx > 0:
            progress_pct = current_question_idx / len(required_fields)
            st.progress(progress_pct)
            st.markdown(f"<p class='progress-text'>Question {current_question_idx} of {len(required_fields)}</p>", unsafe_allow_html=True)
        
        # Questions Display
        st.markdown("<div class='questions-section'>", unsafe_allow_html=True)
        
        if next_required_field and not already_submitted_today:
            current_field = next_required_field
            options = get_field_options(current_field)
            question = get_question_for_field(current_field)
            
            st.markdown(f"<div class='question-text'>{question}</div>", unsafe_allow_html=True)
            
            if current_field == "Age":
                age_value = st.number_input(
                    "Age",
                    min_value=10,
                    max_value=100,
                    step=1,
                    value=_format_age_value(answers.get(current_field, 21)),
                    format="%d",
                    key=f"num_{current_field}",
                )
                if st.button("Save Age", key=f"save_{current_field}", width="stretch"):
                    answers[current_field] = str(int(age_value))
                    if current_user_id and current_field in STATIC_USER_FIELDS:
                        save_user_static_answer_if_missing(current_user_id, current_field, answers[current_field])
                    st.session_state["last_answers"] = answers
                    st.rerun()
            elif current_field == "CGPA":
                cgpa_value = st.number_input(
                    "CGPA",
                    min_value=0.0,
                    max_value=4.0,
                    step=0.01,
                    value=_format_cgpa_value(answers.get(current_field, 3.0)),
                    format="%.2f",
                    key=f"num_{current_field}",
                )
                if st.button("Save CGPA", key=f"save_{current_field}", width="stretch"):
                    answers[current_field] = f"{float(cgpa_value):.2f}".rstrip("0").rstrip(".")
                    if current_user_id and current_field in STATIC_USER_FIELDS:
                        save_user_static_answer_if_missing(current_user_id, current_field, answers[current_field])
                    st.session_state["last_answers"] = answers
                    st.rerun()
            elif options:
                for idx, option in enumerate(options):
                    if st.button(option, key=f"btn_{current_field}_{idx}", width="stretch"):
                        # Store the exact UI option the user clicked (no normalization here).
                        # Model-side normalization happens in scripts/integrated_model_inference.py.
                        answers[current_field] = option

                        # Save first baseline response permanently for logged-in users.
                        if current_user_id and current_field in STATIC_USER_FIELDS:
                            save_user_static_answer_if_missing(current_user_id, current_field, answers[current_field])

                        st.session_state["last_answers"] = answers
                        st.rerun()
            else:
                user_input = st.text_input(f"Your answer:", key=f"input_{current_field}")
                if user_input:
                    answers[current_field] = user_input
                    st.session_state["last_answers"] = answers
                    st.rerun()
        elif not already_submitted_today:
            st.markdown("<div style='text-align: center; padding: 30px 0;'><h3 style='color: #3F2456;'>✨ All set!</h3><p style='color: #7A6B8F;'>Click below to analyze your wellbeing</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; padding: 30px 0;'><h3 style='color: #3F2456;'>📅 Daily survey completed</h3><p style='color: #7A6B8F;'>Your next survey unlocks tomorrow.</p></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analyze Button
        st.markdown("<div class='analyze-section'>", unsafe_allow_html=True)
        if st.button("🔍 Analyze My Results", key="analyze_btn", width="stretch", disabled=already_submitted_today):
            if not all_required_answered(answers, required_fields):
                st.warning("⚠️ Please answer all questions first!")
            else:
                try:
                    from pathlib import Path
                    from scripts.integrated_model_inference import integrated_predict

                    if current_user_id:
                        save_user_static_answers(current_user_id, answers)

                    integrated_output = integrated_predict(answers, Path(APP_DIR))
                    cluster = 0

                    prediction = compact_model_outputs(integrated_output)

                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_cluster"] = cluster
                    st.session_state["show_results"] = True

                    save_user_response_to_json(answers, prediction, cluster, user_id=st.session_state.get("current_user_id"))
                    if st.session_state.get("current_user_id"):
                        current_user_id = str(st.session_state.get("current_user_id") or "")
                        saved_to_sql, save_message = save_user_daily_input_to_sql(
                            user_id=current_user_id,
                            answers=answers,
                            prediction=prediction,
                            cluster=cluster,
                        )
                        if saved_to_sql:
                            st.success("✅ Daily input saved to your account history.")
                        else:
                            st.warning(f"⚠️ {save_message}")
                    else:
                        st.info("ℹ️ Log in to save daily inputs to your account history.")
                except Exception as exc:
                    st.warning(f"⚠️ Unable to run integrated inference: {exc}")

        if st.button("🔄 Start New Assessment", key="reset_assessment_btn", width="stretch"):
            reset_survey_state()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Results Display
        if st.session_state.get("show_results"):
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            
            prediction = st.session_state.get("last_prediction")

            st.markdown("<h3 style='text-align: center;'>📊 Random Forest Output</h3>", unsafe_allow_html=True)
            st.caption("Survey analysis uses your trained random forest model only.")

            burnout_class, score_label, score_value, selected_details = get_selected_model_output(prediction, "Random Forest")
            score_display = f"{score_value:.1%}"

            col_model_a, col_model_b = st.columns(2)
            with col_model_a:
                st.metric("Burnout class type", burnout_class)
            with col_model_b:
                st.metric(score_label, score_display)

            with st.expander("View model details"):
                st.json(selected_details)
            
            st.markdown("</div>", unsafe_allow_html=True)

        if current_user_id:
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            render_user_progress_section(current_user_id)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        
    
    # ========== RIGHT COLUMN: Authentication ==========
    with right_col:
        if st.button("📊 Model Analysis", key="analysis_btn", width="stretch"):
            st.session_state["auth_page"] = "model_analysis"
            st.rerun()
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        if st.session_state.get("current_user_id"):
            st.markdown("<div class='auth-header'>Logged In</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='auth-subtext'>{st.session_state.get('current_user_name')}<br/>{st.session_state.get('current_user_email')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button("👤 Profile", key="profile_btn", width="stretch"):
                st.session_state["auth_page"] = "profile"
                st.rerun()
            if is_admin_user():
                if st.button("🛠️ Admin View", key="admin_btn_logged_in", width="stretch"):
                    st.session_state["auth_page"] = "admin"
                    st.rerun()
            if st.button("🚪 Log Out", key="logout_main_btn", width="stretch"):
                st.session_state["current_user_id"] = None
                st.session_state["current_user_email"] = None
                st.session_state["current_user_name"] = None
                st.session_state["current_user_phone_number"] = None
                st.rerun()
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-subtext'>Supabase history enabled</div>", unsafe_allow_html=True)
            if st.session_state.get("last_data_save_error"):
                st.caption(f"Data save note: {st.session_state.get('last_data_save_error')}")
            if st.session_state.get("last_supabase_sync_error"):
                st.caption(f"Supabase sync note: {st.session_state.get('last_supabase_sync_error')}")
        else:
        
            st.markdown("<div class='auth-header'>Join AuraCheck</div>", unsafe_allow_html=True)
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            
            if st.button("⭐ Sign Up", key="signup_btn", width="stretch"):
                st.session_state["auth_page"] = "signup"
                st.rerun()
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button("📝 Log In", key="login_btn", width="stretch"):
                st.session_state["auth_page"] = "login"
                st.rerun()
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-subtext'>Save your progress &<br/>track your journey</div>", unsafe_allow_html=True)
    
    # --- CLOSE CONTENT PLACEHOLDER ---
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
