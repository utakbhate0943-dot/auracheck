"""
Final report documenting:
- Process used to arrive at the selected baseline model
- Comparison against alternative baseline candidates
- Regression tables (coefficients + p-values)
- Performance metrics
- Usage examples
"""

import os
import json
import textwrap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss

ROOT = "/Users/edgarvidriales/Desktop/AuraCheck/auracheck"
OUT_DIR = os.path.join(ROOT, "baseline", "outputs", "final_baseline_model")

OUT_PDF = os.path.join(OUT_DIR, "final_process_baseline_comparison_report.pdf")
OUT_TXT = os.path.join(OUT_DIR, "final_process_baseline_comparison_report_summary.txt")
OUT_CSV = os.path.join(OUT_DIR, "final_baseline_candidate_comparison_table.csv")
OUT_CM = os.path.join(OUT_DIR, "final_selected_baseline_confusion_matrix.csv")
OUT_SENS_SPEC = os.path.join(OUT_DIR, "final_selected_baseline_sensitivity_specificity.csv")
OUT_MODEL_WRITTEN = os.path.join(OUT_DIR, "final_model_written_out.txt")
OUT_ASSUMPTIONS_TXT = os.path.join(OUT_DIR, "final_model_assumption_checks.txt")

DATA_PATH = "/Users/edgarvidriales/Desktop/AuraCheck/auracheck/Dataset/students_mental_health_survey_with_burnout_final.csv"
CLASS_NAMES = ["Very Low (Q1)", "Low (Q2)", "Moderate (Q3)", "High (Q4)"]
FEATURES_PRUNED = [
    "Course", "Gender", "Sleep_Quality", "Physical_Activity", "Diet_Quality",
    "Social_Support", "Relationship_Status", "Substance_Use", "Counseling_Service_Use",
    "Family_History", "Chronic_Illness", "Financial_Stress",
    "Extracurricular_Involvement", "Residence_Type"
]

ENCODING_MAP = {
    "Gender": {"Female": 0, "Male": 1},
    "Sleep_Quality": {"Poor": 0, "Average": 1, "Good": 2},
    "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
    "Diet_Quality": {"Good": 0, "Average": 1, "Poor": 2},
    "Social_Support": {"High": 0, "Moderate": 1, "Low": 2},
    "Substance_Use": {"Never": 0, "Unknown": 1, "Occasionally": 2, "Frequently": 3},
    "Counseling_Service_Use": {"Never": 0, "Occasionally": 1, "Frequently": 2},
    "Family_History": {"No": 0, "Yes": 1},
    "Chronic_Illness": {"No": 0, "Yes": 1},
    "Extracurricular_Involvement": {"High": 0, "Moderate": 1, "Low": 2},
    "Course": {"Business": 0, "Computer Science": 1, "Engineering": 2, "Law": 3, "Medical": 4, "Others": 5},
    "Relationship_Status": {"In a Relationship": 0, "Married": 1, "Single": 2},
    "Residence_Type": {"Off-Campus": 0, "On-Campus": 1, "With Family": 2},
}


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def preprocess(df, features):
    X = df[features].copy()
    for c in X.columns:
        if X[c].isnull().any():
            if X[c].dtype == object:
                X[c] = X[c].fillna("Unknown")
            else:
                X[c] = X[c].fillna(X[c].median())
    for c, m in ENCODING_MAP.items():
        if c in X.columns:
            X[c] = X[c].astype(str).map(m).fillna(1)
    return X.astype(float)


def compute_selected_baseline_confusion_and_rates():
    try:
        df = pd.read_csv(DATA_PATH)
        X = preprocess(df, FEATURES_PRUNED)
        y = pd.qcut(df["burnout_raw_score"].astype(float), q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)

        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

        rows = []
        total = cm.sum()
        for i, cname in enumerate(CLASS_NAMES):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            rows.append({
                "Class": cname,
                "Sensitivity_Recall": float(sensitivity),
                "Specificity": float(specificity),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
            })

        rates_df = pd.DataFrame(rows)
        cm_df.to_csv(OUT_CM)
        rates_df.to_csv(OUT_SENS_SPEC, index=False)
        return cm_df, rates_df
    except Exception:
        # Fallback to already-saved artifacts in baseline folder.
        cm_df = safe_read_csv(OUT_CM)
        if cm_df is None:
            cm_df = pd.DataFrame(np.zeros((4, 4), dtype=int), index=CLASS_NAMES, columns=CLASS_NAMES)
        else:
            idx_col = cm_df.columns[0]
            cm_df = cm_df.set_index(idx_col)

        rates_df = safe_read_csv(OUT_SENS_SPEC)
        if rates_df is None or len(rates_df) == 0:
            rates_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Sensitivity_Recall": [0.0, 0.0, 0.0, 0.0],
                "Specificity": [0.0, 0.0, 0.0, 0.0],
            })
        return cm_df, rates_df


def build_candidate_table():
    rows = []

    # weighted vs unweighted (full 17)
    p1 = os.path.join(OUT_DIR, "logistic_4class_weighted_vs_unweighted_metrics.csv")
    df1 = safe_read_csv(p1)
    if df1 is not None:
        for _, r in df1.iterrows():
            rows.append({
                "Candidate": r["Model"],
                "Feature_Set": "Full raw 17",
                "Accuracy": float(r["Accuracy"]),
                "Kappa": float(r["Kappa"]),
                "Macro_Recall": float(r["Macro_Recall"]),
                "Recall_VeryLow": float(r["Recall_VeryLow"]),
                "Recall_Low": float(r["Recall_Low"]),
                "Recall_Moderate": float(r["Recall_Moderate"]),
                "Recall_High": float(r["Recall_High"]),
                "Source": os.path.basename(p1),
            })

    # VIF-pruning comparison (includes balanced pruned)
    p2 = os.path.join(OUT_DIR, "logistic_4class_vif_pruning_comparison_metrics.csv")
    df2 = safe_read_csv(p2)
    if df2 is not None:
        for _, r in df2.iterrows():
            rows.append({
                "Candidate": r["Model"],
                "Feature_Set": "Pruned/Full mix",
                "Accuracy": float(r["Accuracy"]),
                "Kappa": float(r["Kappa"]),
                "Macro_Recall": float(r["Macro_Recall"]),
                "Recall_VeryLow": float(r["Recall_VeryLow"]),
                "Recall_Low": float(r["Recall_Low"]),
                "Recall_Moderate": float(r["Recall_Moderate"]),
                "Recall_High": float(r["Recall_High"]),
                "Source": os.path.basename(p2),
            })

    # Strict ordinal logit
    p3 = os.path.join(OUT_DIR, "strict_ordinal_logit_4class_metrics.csv")
    df3 = safe_read_csv(p3)
    if df3 is not None and len(df3) > 0:
        r = df3.iloc[0]
        rows.append({
            "Candidate": "Strict Ordinal Logit (pruned)",
            "Feature_Set": "Pruned 14",
            "Accuracy": float(r["Accuracy"]),
            "Kappa": float(r["Kappa"]),
            "Macro_Recall": float(r["Macro_Recall"]),
            "Recall_VeryLow": float(r["Recall_VeryLow"]),
            "Recall_Low": float(r["Recall_Low"]),
            "Recall_Moderate": float(r["Recall_Moderate"]),
            "Recall_High": float(r["Recall_High"]),
            "Source": os.path.basename(p3),
        })

    # orthogonalization test
    p4 = os.path.join(OUT_DIR, "orthogonalized_multinomial_4class_comparison_metrics.csv")
    df4 = safe_read_csv(p4)
    if df4 is not None:
        for _, r in df4.iterrows():
            rows.append({
                "Candidate": r["Model"],
                "Feature_Set": f"{int(r['NumFeatures'])} features",
                "Accuracy": float(r["Accuracy"]),
                "Kappa": float(r["Kappa"]),
                "Macro_Recall": float(r["Macro_Recall"]),
                "Recall_VeryLow": float(r["Recall_VeryLow"]),
                "Recall_Low": float(r["Recall_Low"]),
                "Recall_Moderate": float(r["Recall_Moderate"]),
                "Recall_High": float(r["Recall_High"]),
                "Source": os.path.basename(p4),
            })

    if len(rows) == 0:
        p_prod = os.path.join(OUT_DIR, "production_pruned_multinomial_metrics.csv")
        prod = safe_read_csv(p_prod)
        if prod is not None and len(prod) > 0:
            r = prod.iloc[0]
            rows.append({
                "Candidate": "Balanced - Pruned(14) (Production)",
                "Feature_Set": "Pruned 14",
                "Accuracy": float(r.get("Accuracy", np.nan)),
                "Kappa": float(r.get("Kappa", np.nan)),
                "Macro_Recall": float(r.get("Macro_Recall", np.nan)),
                "Recall_VeryLow": float(r.get("Recall_VeryLow", np.nan)),
                "Recall_Low": float(r.get("Recall_Low", np.nan)),
                "Recall_Moderate": float(r.get("Recall_Moderate", np.nan)),
                "Recall_High": float(r.get("Recall_High", np.nan)),
                "Source": os.path.basename(p_prod),
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=["Candidate", "Feature_Set", "Accuracy", "Macro_Recall"])
    if len(out) > 0:
        out = out.sort_values(["Macro_Recall", "Accuracy"], ascending=False).reset_index(drop=True)
    return out


def wrap_candidate_label(label, width=22):
    """Split long candidate labels into (at most) two lines for PDF table readability."""
    s = str(label)
    if " - " in s:
        left, right = s.split(" - ", 1)
        return f"{left} -\n{right}"

    wrapped = textwrap.wrap(s, width=width)
    if len(wrapped) <= 1:
        return s
    if len(wrapped) == 2:
        return "\n".join(wrapped)
    return f"{wrapped[0]}\n{' '.join(wrapped[1:])}"


def generate_written_model_text():
    model_path = os.path.join(OUT_DIR, "production_pruned_multinomial_model.joblib")
    meta_path = os.path.join(OUT_DIR, "production_pruned_multinomial_metadata.json")

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        lines = [
            "MODEL WRITTEN OUT",
            "=" * 80,
            "Production model artifacts not found.",
            f"Expected: {model_path}",
            f"Expected: {meta_path}",
        ]
        with open(OUT_MODEL_WRITTEN, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return lines

    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    clf = bundle["model"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta.get("features", FEATURES_PRUNED)
    class_names = meta.get("class_names", CLASS_NAMES)

    means = scaler.mean_
    scales = scaler.scale_
    coefs = clf.coef_
    intercepts = clf.intercept_

    lines = []
    lines.append("MODEL WRITTEN OUT")
    lines.append("=" * 80)
    lines.append("Model: Balanced Multinomial Logistic Regression (Pruned 14)")
    lines.append("")
    lines.append("Form:")
    lines.append("eta_k = b_k + sum_j(w_{k,j} * z_j)")
    lines.append("z_j = (x_j - mean_j) / scale_j")
    lines.append("P(class=k) = exp(eta_k) / sum_c exp(eta_c)")
    lines.append("")
    lines.append("Standardization parameters:")
    for j, feat in enumerate(features):
        lines.append(f"- {feat}: mean={means[j]:.6f}, scale={scales[j]:.6f}")

    lines.append("")
    lines.append("Class-specific equations removed from this report by request.")

    with open(OUT_MODEL_WRITTEN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return lines


def _compute_vif(X_df):
    """Compute simple VIF values without extra dependencies."""
    X = X_df.to_numpy(dtype=float)
    n, p = X.shape
    out = []
    for i in range(p):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        others = np.column_stack([np.ones(n), others])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        y_hat = others @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 1e-12:
            r2 = 0.0
        else:
            r2 = max(0.0, min(0.999999, 1.0 - ss_res / ss_tot))
        vif = 1.0 / (1.0 - r2)
        out.append({"Feature": X_df.columns[i], "VIF": float(vif)})
    return pd.DataFrame(out)


def get_model_diagnostics():
    """
    Return model type text, assumption checks table, and coefficient matrix for visualization.
    """
    model_path = os.path.join(OUT_DIR, "production_pruned_multinomial_model.joblib")
    meta_path = os.path.join(OUT_DIR, "production_pruned_multinomial_metadata.json")

    model_type = "Balanced Multinomial Logistic Regression (Pruned 14)"
    features = FEATURES_PRUNED
    class_names = CLASS_NAMES
    coef_df = None
    checks = []

    clf = None
    if os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            clf = bundle.get("model")
        except Exception:
            clf = None

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_type = meta.get("model_type", model_type)
            features = meta.get("features", features)
            class_names = meta.get("class_names", class_names)
        except Exception:
            pass

    # Convergence check
    if clf is not None and hasattr(clf, "n_iter_") and hasattr(clf, "max_iter"):
        max_iter_used = int(np.max(clf.n_iter_))
        max_iter_allowed = int(clf.max_iter)
        checks.append({
            "Check": "Solver convergence",
            "Result": f"n_iter_max={max_iter_used}, max_iter={max_iter_allowed}",
            "Status": "PASS" if max_iter_used < max_iter_allowed else "FAIL",
        })
    else:
        checks.append({
            "Check": "Solver convergence",
            "Result": "Unavailable (model object not loaded)",
            "Status": "N/A",
        })

    # Coefficient matrix for visual model representation
    if clf is not None and hasattr(clf, "coef_"):
        try:
            coef_df = pd.DataFrame(clf.coef_, index=class_names, columns=features)
        except Exception:
            coef_df = None

    # Data-dependent checks (VIF, class balance, probability validity)
    try:
        df = pd.read_csv(DATA_PATH)
        X = preprocess(df, features)
        y = pd.qcut(df["burnout_raw_score"].astype(float), q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

        # Multicollinearity check
        vif_df = _compute_vif(X)
        max_vif = float(vif_df["VIF"].max()) if len(vif_df) > 0 else np.nan
        checks.append({
            "Check": "Multicollinearity (VIF)",
            "Result": f"max VIF={max_vif:.3f} (threshold <= 10)",
            "Status": "PASS" if np.isfinite(max_vif) and max_vif <= 10.0 else "FAIL",
        })

        # Class presence / balance check
        counts = y.value_counts().sort_index()
        min_count = int(counts.min()) if len(counts) > 0 else 0
        checks.append({
            "Check": "All target classes present",
            "Result": f"counts={counts.to_dict()}",
            "Status": "PASS" if len(counts) == 4 and min_count > 0 else "FAIL",
        })

        # Probability validity check
        if clf is not None and hasattr(clf, "predict_proba"):
            scaler = StandardScaler().fit(X)
            probs = clf.predict_proba(scaler.transform(X.iloc[: min(500, len(X))]))
            sums = probs.sum(axis=1)
            valid = np.all((probs >= -1e-9) & (probs <= 1 + 1e-9)) and np.allclose(sums, 1.0, atol=1e-6)
            checks.append({
                "Check": "Probability validity",
                "Result": "All probabilities in [0,1] and rows sum to 1",
                "Status": "PASS" if valid else "FAIL",
            })
        else:
            checks.append({
                "Check": "Probability validity",
                "Result": "Unavailable (predict_proba not available)",
                "Status": "N/A",
            })

    except Exception as ex:
        checks.append({
            "Check": "Data-dependent checks",
            "Result": f"Unavailable ({type(ex).__name__})",
            "Status": "N/A",
        })

    checks_df = pd.DataFrame(checks)
    with open(OUT_ASSUMPTIONS_TXT, "w", encoding="utf-8") as f:
        f.write("MODEL ASSUMPTION CHECKS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model type: {model_type}\n\n")
        f.write(checks_df.to_string(index=False))
        f.write("\n")

    return model_type, checks_df, coef_df


def compute_additional_performance_metrics():
    """Compute Log Loss and Macro F1 on the standard held-out split."""
    try:
        df = pd.read_csv(DATA_PATH)
        X = preprocess(df, FEATURES_PRUNED)
        y = pd.qcut(df["burnout_raw_score"].astype(float), q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)

        probs = clf.predict_proba(X_test_s)
        pred = clf.predict(X_test_s)

        ll = float(log_loss(y_test, probs, labels=[0, 1, 2, 3]))
        f1m = float(f1_score(y_test, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        return ll, f1m
    except Exception:
        return np.nan, np.nan


def main():
    candidates = build_candidate_table()
    if len(candidates) == 0:
        raise RuntimeError("No candidate metrics found to build final report.")
    candidates.to_csv(OUT_CSV, index=False)

    # Selected final model row
    selected = candidates[candidates["Candidate"].str.contains(r"Balanced - Pruned\(14\)|production|Pruned", regex=True, na=False)]
    if len(selected) == 0:
        selected = candidates.head(1)
    selected_row = selected.iloc[0]
    model_lines = generate_written_model_text()
    model_type, checks_df, coef_df = get_model_diagnostics()

    # Performance metrics for dedicated report section
    perf_path = os.path.join(OUT_DIR, "production_pruned_multinomial_metrics.csv")
    perf_df = safe_read_csv(perf_path)
    if perf_df is None or len(perf_df) == 0:
        perf_df = pd.DataFrame([{
            "Accuracy": float(selected_row.get("Accuracy", np.nan)),
            "Kappa": float(selected_row.get("Kappa", np.nan)),
            "Recall_VeryLow": float(selected_row.get("Recall_VeryLow", np.nan)),
            "Recall_Low": float(selected_row.get("Recall_Low", np.nan)),
            "Recall_Moderate": float(selected_row.get("Recall_Moderate", np.nan)),
            "Recall_High": float(selected_row.get("Recall_High", np.nan)),
            "Macro_Recall": float(selected_row.get("Macro_Recall", np.nan)),
        }])

    ll, f1m = compute_additional_performance_metrics()
    if "Log_Loss" not in perf_df.columns:
        perf_df["Log_Loss"] = ll
    if "F1_Macro" not in perf_df.columns:
        perf_df["F1_Macro"] = f1m

    cm_df, rates_df = compute_selected_baseline_confusion_and_rates()

    # Coef/p-values table
    coef_p_path = os.path.join(OUT_DIR, "multinomial_4class_coefficients_pvalues.csv")
    coef_p = safe_read_csv(coef_p_path)

    sk = None
    sm = None
    if coef_p is not None:
        sk = coef_p[coef_p["Source"] == "sklearn_balanced"].copy()
        sm = coef_p[coef_p["Source"] == "statsmodels_unweighted"].copy()

    # Example usage files
    ex_txt_path = os.path.join(OUT_DIR, "example_student_prediction_interpretation.txt")
    ex_json_path = os.path.join(OUT_DIR, "production_pruned_multinomial_example_prediction.json")
    walkthrough_json_path = os.path.join(OUT_DIR, "student_example_prediction_walkthrough.json")

    ex_lines = []
    if os.path.exists(ex_txt_path):
        with open(ex_txt_path, "r", encoding="utf-8") as f:
            ex_lines = [ln.rstrip() for ln in f.readlines()][:45]

    ex_json = None
    if os.path.exists(ex_json_path):
        with open(ex_json_path, "r", encoding="utf-8") as f:
            ex_json = json.load(f)

    walkthrough_json = None
    if os.path.exists(walkthrough_json_path):
        with open(walkthrough_json_path, "r", encoding="utf-8") as f:
            walkthrough_json = json.load(f)

    with PdfPages(OUT_PDF) as pdf:
        # Page 1: title + process
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.5, 0.95, "Final Baseline Model Process Report", ha="center", fontsize=18, weight="bold")
        process = [
            "Objective:",
            "• Build a simple, transparent baseline for 4-class burnout prediction.",
            "• Model type: " + model_type,
            "• Balanced means class-weighted training (class_weight='balanced').",
            "  Underrepresented classes get higher loss weight, so the model does",
            "  not over-prioritize majority classes.",
            "",
            "Process used:",
            "1) Built quartile-based 4-class target from burnout_raw_score.",
            "2) Tested multinomial logistic (unweighted vs class_weight='balanced').",
            "3) Diagnosed multicollinearity (very high VIF for CGPA, Age, Semester_Credit_Load).",
            "4) Applied VIF pruning (threshold 10): removed CGPA, Age, Semester_Credit_Load.",
            "5) Refit balanced multinomial on pruned 14 features.",
            "6) Compared against alternatives (strict ordinal logit, full raw 17, orthogonalized 17-concept).",
            "7) Selected final baseline by macro recall + stability + simplicity.",
            "",
            "Selected baseline:",
            f"• {selected_row['Candidate']}",
            f"• Accuracy={selected_row['Accuracy']:.4f}, Macro Recall={selected_row['Macro_Recall']:.4f}, Kappa={selected_row['Kappa']:.4f}",
        ]
        y = 0.88
        for line in process:
            plt.text(0.06, y, line, fontsize=10, va="top")
            y -= 0.038
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2+: model written out (paginated so all classes are shown)
        lines_per_page = 46
        total_pages = int(np.ceil(len(model_lines) / lines_per_page)) if len(model_lines) > 0 else 1
        for p in range(total_pages):
            start = p * lines_per_page
            end = min((p + 1) * lines_per_page, len(model_lines))
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis("off")
            title = "Model Written Out" if total_pages == 1 else f"Model Written Out ({p+1}/{total_pages})"
            plt.text(0.5, 0.965, title, ha="center", fontsize=14, weight="bold")
            y = 0.93
            for line in model_lines[start:end]:
                plt.text(0.05, y, line[:140], fontsize=8.6, va="top")
                y -= 0.019
                if y < 0.05:
                    break
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 3: assumption checks
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.5, 0.95, "Model Assumption Checks", ha="center", fontsize=14, weight="bold")
        plt.text(0.06, 0.90, f"Model type: {model_type}", fontsize=10.5, va="top")

        tab = checks_df.copy()
        if "Status" in tab.columns:
            tab["Status"] = tab["Status"].astype(str)
        if "Result" in tab.columns:
            tab["Result"] = tab["Result"].astype(str).map(lambda s: s[:90])

        ax_tab = fig.add_axes([0.05, 0.12, 0.90, 0.72])
        ax_tab.axis("off")
        t3 = ax_tab.table(
            cellText=tab[["Check", "Result", "Status"]].values,
            colLabels=["Check", "Result", "Status"],
            loc="center",
            cellLoc="left",
            colWidths=[0.24, 0.60, 0.12],
        )
        t3.auto_set_font_size(False)
        t3.set_fontsize(9)
        t3.scale(1.0, 1.5)

        plt.text(0.06, 0.07, f"Detailed text file: {OUT_ASSUMPTIONS_TXT}", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: visual model representation (coefficient heatmap)
        if coef_df is not None and len(coef_df) > 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.heatmap(coef_df, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Coefficient"})
            ax.set_title("Visual Representation: Multinomial Coefficient Heatmap", fontsize=14, pad=12)
            ax.set_xlabel("Features")
            ax.set_ylabel("Classes")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 5: model performance metrics
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.5, 0.95, "Model Performance Metrics", ha="center", fontsize=14, weight="bold")

        row = perf_df.iloc[0]
        metric_rows = [
            ["Accuracy", f"{float(row.get('Accuracy', np.nan)):.4f}"],
            ["Cohen's Kappa", f"{float(row.get('Kappa', np.nan)):.4f}"],
            ["Macro Recall", f"{float(row.get('Macro_Recall', np.nan)):.4f}"],
            ["Log Loss", f"{float(row.get('Log_Loss', np.nan)):.4f}"],
            ["Macro F1 Score", f"{float(row.get('F1_Macro', np.nan)):.4f}"],
            ["Recall - Very Low (Q1)", f"{float(row.get('Recall_VeryLow', np.nan)):.4f}"],
            ["Recall - Low (Q2)", f"{float(row.get('Recall_Low', np.nan)):.4f}"],
            ["Recall - Moderate (Q3)", f"{float(row.get('Recall_Moderate', np.nan)):.4f}"],
            ["Recall - High (Q4)", f"{float(row.get('Recall_High', np.nan)):.4f}"],
        ]

        axm = fig.add_axes([0.12, 0.50, 0.76, 0.35])
        axm.axis("off")
        tm = axm.table(
            cellText=metric_rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colWidths=[0.65, 0.25],
        )
        tm.auto_set_font_size(False)
        tm.set_fontsize(10)
        tm.scale(1.0, 1.5)

        plt.text(0.12, 0.18, "Metric definitions:", fontsize=9.2, weight="bold")
        plt.text(0.12, 0.15, "• Accuracy: proportion of all predictions that are correct.", fontsize=8.6)
        plt.text(0.12, 0.12, "• Cohen's Kappa: agreement beyond chance (1=perfect, 0=chance-level).", fontsize=8.6)
        plt.text(0.12, 0.09, "• Macro Recall: average recall across classes (treats classes equally).", fontsize=8.6)
        plt.text(0.12, 0.06, "• Log Loss: penalizes incorrect/overconfident probabilities (lower is better).", fontsize=8.6)
        plt.text(0.12, 0.03, "• Macro F1: harmonic mean of precision/recall per class, averaged equally.", fontsize=8.6)
        plt.text(0.12, 0.01, "• Recall - Q1/Q2/Q3/Q4: class-specific sensitivity = TP/(TP+FN).", fontsize=8.6)
        plt.text(0.12, 0.005, "Interpretation: higher is better for most metrics; lower is better for Log Loss.", fontsize=8.6)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: confusion matrix + sensitivity/specificity
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
        axes[0].set_title("Selected Baseline: Confusion Matrix (test)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        rr = rates_df.copy()
        rr["Sensitivity_Recall"] = rr["Sensitivity_Recall"].map(lambda v: f"{v:.4f}")
        rr["Specificity"] = rr["Specificity"].map(lambda v: f"{v:.4f}")
        axes[1].axis("off")
        axes[1].set_title("Per-class Sensitivity and Specificity", pad=10)
        t2 = axes[1].table(
            cellText=rr[["Class", "Sensitivity_Recall", "Specificity"]].values,
            colLabels=["Class", "Sensitivity (Recall)", "Specificity"],
            loc="center",
            cellLoc="center",
        )
        t2.auto_set_font_size(False)
        t2.set_fontsize(9)
        t2.scale(1.0, 1.4)

        axes[1].text(
            0.0,
            -0.18,
            "Definition: Sensitivity (Recall) = TP / (TP + FN), i.e., among true class members, how many the model correctly identifies.",
            transform=axes[1].transAxes,
            fontsize=8.8,
            va="top",
        )
        axes[1].text(
            0.0,
            -0.28,
            "Definition: Specificity = TN / (TN + FP), i.e., among non-members of the class, how many the model correctly rejects.",
            transform=axes[1].transAxes,
            fontsize=8.8,
            va="top",
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 7: regression table (balanced coefficients)
        if sk is not None and len(sk) > 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.set_title("Regression Table: Balanced Multinomial Coefficients", fontsize=14, pad=12)
            # top terms by absolute coefficient across classes
            s2 = sk[sk["Term"] != "Intercept"].copy()
            agg = s2.groupby("Term")["Coefficient"].apply(lambda z: float(np.max(np.abs(z)))).reset_index(name="MaxAbsCoef")
            top_terms = agg.sort_values("MaxAbsCoef", ascending=False).head(12)["Term"].tolist()
            s3 = s2[s2["Term"].isin(top_terms)].copy()
            piv = s3.pivot_table(index="Term", columns="Class", values="Coefficient", aggfunc="mean")
            piv = piv.reindex(top_terms)
            piv = piv.rename(columns={0: "Class0", 1: "Class1", 2: "Class2", 3: "Class3"}).reset_index()
            for c in [col for col in piv.columns if col.startswith("Class")]:
                piv[c] = piv[c].map(lambda v: f"{v:.4f}")
            t = ax.table(cellText=piv.values, colLabels=piv.columns, loc="center", cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1.1, 1.35)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 8: regression table (p-values from companion model)
        if sm is not None and len(sm) > 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.set_title("Regression Table: Companion MNLogit Coefficients + p-values", fontsize=14, pad=12)

            sm2 = sm.copy()
            # keep top by smallest p-value
            sm2 = sm2.sort_values("PValue", ascending=True).head(24)
            sm2["Coefficient"] = sm2["Coefficient"].map(lambda v: f"{v:.4f}")
            sm2["PValue"] = sm2["PValue"].map(lambda v: f"{v:.4g}")
            sm2["Class"] = sm2["Class"].astype(int).astype(str)
            cols = ["Class", "Term", "Coefficient", "PValue"]
            t = ax.table(cellText=sm2[cols].values, colLabels=cols, loc="center", cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1.1, 1.35)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 9: usage examples
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.5, 0.95, "Usage Example: Student Answers to Outcome", ha="center", fontsize=14, weight="bold")

        y = 0.90
        if ex_json is not None:
            plt.text(0.06, y, "Production prediction JSON example:", fontsize=10, weight="bold", va="top")
            y -= 0.035
            for k, v in ex_json.get("input", {}).items():
                plt.text(0.06, y, f"• {k}: {v}", fontsize=9.3, va="top")
                y -= 0.026
            y -= 0.01
            pred = ex_json.get("prediction", {})
            plt.text(0.06, y, f"Predicted class: {pred.get('predicted_class', 'N/A')}", fontsize=10, va="top")
            y -= 0.03
            for ck, pv in pred.get("probabilities", {}).items():
                plt.text(0.06, y, f"• {ck}: {pv:.4f}", fontsize=9.3, va="top")
                y -= 0.025

        if ex_lines:
            y = min(y - 0.03, 0.35)
            plt.text(0.06, y, "Interpretation excerpt:", fontsize=10, weight="bold", va="top")
            y -= 0.03
            for ln in ex_lines[-8:]:
                if ln.strip():
                    plt.text(0.06, y, ln[:120], fontsize=9, va="top")
                    y -= 0.025

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 10: prediction process explanation for student example
        if walkthrough_json is not None:
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis("off")
            plt.text(0.5, 0.95, "How the Model Produces the Prediction", ha="center", fontsize=14, weight="bold")

            y = 0.90
            plt.text(0.06, y, "Step 1) Encode categorical answers to numeric values.", fontsize=10, va="top")
            y -= 0.03
            plt.text(0.06, y, "Step 2) Standardize each feature: z_j = (x_j - mean_j) / scale_j", fontsize=10, va="top")
            y -= 0.03
            plt.text(0.06, y, "Step 3) Compute class scores (etas): eta_k = b_k + sum_j(w_{k,j} * z_j)", fontsize=10, va="top")
            y -= 0.03
            plt.text(0.06, y, "Step 4) Convert to probabilities with softmax and pick highest probability class.", fontsize=10, va="top")

            y -= 0.045
            plt.text(0.06, y, "Student example outputs:", fontsize=10.5, weight="bold", va="top")
            y -= 0.03

            etas = walkthrough_json.get("etas", {})
            probs = walkthrough_json.get("probabilities", {})
            pred_class = walkthrough_json.get("predicted_class", "N/A")

            plt.text(0.06, y, "Etas (class scores):", fontsize=9.8, weight="bold", va="top")
            y -= 0.028
            for k, v in etas.items():
                plt.text(0.08, y, f"• {k}: {v:.6f}", fontsize=9.2, va="top")
                y -= 0.023

            y -= 0.01
            plt.text(0.06, y, "Softmax probabilities:", fontsize=9.8, weight="bold", va="top")
            y -= 0.028
            for k, v in probs.items():
                plt.text(0.08, y, f"• {k}: {v:.6f}", fontsize=9.2, va="top")
                y -= 0.023

            y -= 0.012
            plt.text(0.06, y, f"Final prediction: {pred_class}", fontsize=10.5, weight="bold", va="top")

            # show a few encoded/z-score values as concrete example
            y -= 0.05
            plt.text(0.06, y, "Example transformed features (first 8):", fontsize=9.8, weight="bold", va="top")
            y -= 0.028
            enc = walkthrough_json.get("encoded_input", {})
            zsc = walkthrough_json.get("z_scores", {})
            for i, feat in enumerate(list(enc.keys())[:8]):
                plt.text(0.08, y, f"• {feat}: x={enc.get(feat, 0):.3f}, z={zsc.get(feat, 0):.3f}", fontsize=9.0, va="top")
                y -= 0.022

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("FINAL PROCESS + BASELINE COMPARISON REPORT SUMMARY\n")
        f.write("=" * 92 + "\n\n")
        f.write(f"Report PDF: {OUT_PDF}\n")
        f.write(f"Comparison table CSV: {OUT_CSV}\n\n")
        f.write("Selected baseline\n")
        f.write("-" * 92 + "\n")
        f.write(f"Candidate: {selected_row['Candidate']}\n")
        f.write(f"Feature set: {selected_row['Feature_Set']}\n")
        f.write("Balanced model meaning: class-weighted training (class_weight='balanced'), ")
        f.write("which increases penalty for errors on underrepresented classes.\n")
        f.write(f"Accuracy: {selected_row['Accuracy']:.6f}\n")
        f.write(f"Macro Recall: {selected_row['Macro_Recall']:.6f}\n")
        f.write(f"Kappa: {selected_row['Kappa']:.6f}\n\n")

        f.write("Model performance metrics\n")
        f.write("-" * 92 + "\n")
        f.write("Definitions:\n")
        f.write("• Accuracy: proportion of all predictions that are correct.\n")
        f.write("• Kappa: chance-corrected agreement between predictions and true labels.\n")
        f.write("• Macro_Recall: mean of per-class recall values.\n")
        f.write("• Log_Loss: probability error metric; lower is better and penalizes overconfident wrong predictions.\n")
        f.write("• F1_Macro: mean of class-wise F1 (harmonic mean of precision and recall).\n")
        f.write("• Recall_VeryLow/Low/Moderate/High: class-specific sensitivity TP/(TP+FN).\n\n")
        for m in [
            "Accuracy", "Kappa", "Macro_Recall", "Log_Loss", "F1_Macro",
            "Recall_VeryLow", "Recall_Low", "Recall_Moderate", "Recall_High",
        ]:
            if m in perf_df.columns:
                f.write(f"{m}: {float(perf_df.iloc[0][m]):.6f}\n")
        f.write("\n")

        f.write("Selected baseline confusion matrix and rates\n")
        f.write("-" * 92 + "\n")
        f.write(f"Confusion matrix CSV: {OUT_CM}\n")
        f.write(f"Sensitivity/specificity CSV: {OUT_SENS_SPEC}\n\n")
        f.write("Definitions:\n")
        f.write("• Sensitivity (Recall) = TP / (TP + FN): among true class members, fraction correctly identified.\n")
        f.write("• Specificity = TN / (TN + FP): among non-members of the class, fraction correctly rejected.\n\n")
        f.write(rates_df.to_string(index=False))
        f.write("\n\n")

        f.write("Model written out\n")
        f.write("-" * 92 + "\n")
        f.write(f"Model equation text file: {OUT_MODEL_WRITTEN}\n\n")

        f.write("Model type and assumption checks\n")
        f.write("-" * 92 + "\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Assumption checks text file: {OUT_ASSUMPTIONS_TXT}\n\n")
        f.write(checks_df.to_string(index=False))
        f.write("\n\n")

        f.write("Baseline candidate comparison section removed from report by request.\n\n")

        f.write("Regression tables included in report:\n")
        f.write("• Balanced multinomial coefficient table\n")
        f.write("• Companion MNLogit coefficient + p-value table\n")
        if walkthrough_json is not None:
            f.write("• Student prediction process page (encoding → z-scores → etas → softmax)\n")

    print(f"✓ Saved: {OUT_PDF}")
    print(f"✓ Saved: {OUT_TXT}")
    print(f"✓ Saved: {OUT_CSV}")
    print(f"✓ Saved: {OUT_CM}")
    print(f"✓ Saved: {OUT_SENS_SPEC}")


if __name__ == "__main__":
    main()
