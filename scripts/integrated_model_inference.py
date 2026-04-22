"""
Integrated inference for baseline multinomial + unsupervised KMeans + random forest.

Purpose:
- Accept one user input payload (app-style or dataset-style fields).
- Run the trained baseline multinomial model from baseline/outputs/final_baseline_model.
- Run the trained random forest model from ml_randomforest/outputs.
- Run KMeans clustering compatible with Unsupervised/scripts/run_kmeans_unsupervised.py.
- Return a single combined output payload.

Usage examples:
  python scripts/integrated_model_inference.py --input-file Dataset/user_responses.json
  python scripts/integrated_model_inference.py --input-json '{"Course":"Engineering","Gender":"Female"}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


BASELINE_MODEL_PATH = Path("baseline/outputs/final_baseline_model/production_pruned_multinomial_model.joblib")
BASELINE_META_PATH = Path("baseline/outputs/final_baseline_model/production_pruned_multinomial_metadata.json")
RF_MODEL_PATH = Path("ml_randomforest/outputs/best_random_forest_model.joblib")
RF_BUNDLE_PATH = Path("ml_randomforest/outputs/random_forest_inference_bundle.joblib")
RF_SUMMARY_PATH = Path("ml_randomforest/outputs/random_forest_outputs_summary.json")
RF_DATASET_PATH = Path("Dataset/students_mental_health_survey_with_burnout_final.csv")
KMEANS_BUNDLE_PATH = Path("Unsupervised/outputs/baseline_kmeans/kmeans_inference_bundle.joblib")
DATASET_PATH = Path("Dataset/students_mental_health_survey_with_burnout_final.csv")

RF_DROP_COLUMNS = [
    "burnout",
    "burnout_raw_score",
    "burnout_composite_score",
    "burnout_category",
    "Depression_Score",
    "Stress_Level",
    "Anxiety_Score",
    "method1_tertiles",
    "method2_wider",
    "method3_very_wide",
    "method4_manual",
    "method5_manual2",
    "method6_kmeans",
]


def _canonical(s: Any) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _as_float(v: Any, default: float = 5.0) -> float:
    try:
        if isinstance(v, str):
            digits = "".join(ch for ch in v if ch.isdigit() or ch == ".")
            return float(digits) if digits else default
        return float(v)
    except Exception:
        return default


def normalize_from_app_answers(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert app-style answers to dataset/baseline-style values."""

    course_map = {
        "cs": "Computer Science",
        "computerscience": "Computer Science",
        "business": "Business",
        "engineering": "Engineering",
        "medicine": "Medical",
        "medical": "Medical",
        "law": "Law",
        "arts": "Others",
        "science": "Others",
        "others": "Others",
    }
    gender_map = {"male": "Male", "female": "Female", "other": "Female"}
    sleep_map = {
        "poor": "Poor",
        "fair": "Average",
        "average": "Average",
        "good": "Good",
        "verygood": "Good",
        "excellent": "Good",
    }
    pa_map = {
        "none": "Low",
        "minimal": "Low",
        "low": "Low",
        "moderate": "Moderate",
        "active": "High",
        "veryactive": "High",
        "high": "High",
    }
    diet_map = {
        "poor": "Poor",
        "fair": "Average",
        "average": "Average",
        "good": "Good",
        "verygood": "Good",
        "excellent": "Good",
    }
    social_map = {
        "none": "Low",
        "weak": "Low",
        "low": "Low",
        "moderate": "Moderate",
        "good": "High",
        "excellent": "High",
        "high": "High",
    }
    rel_map = {
        "single": "Single",
        "inarelationship": "In a Relationship",
        "married": "Married",
        "complicated": "Single",
        "prefernottosay": "Single",
    }
    sub_map = {
        "none": "Never",
        "never": "Never",
        "occasionally": "Occasionally",
        "regularly": "Frequently",
        "daily": "Frequently",
        "frequently": "Frequently",
    }
    counseling_map = {
        "yesactively": "Frequently",
        "previously": "Occasionally",
        "opentoit": "Occasionally",
        "no": "Never",
        "never": "Never",
        "occasionally": "Occasionally",
        "frequently": "Frequently",
    }
    fh_map = {
        "yesdepression": "Yes",
        "yesanxiety": "Yes",
        "yesother": "Yes",
        "yes": "Yes",
        "no": "No",
        "notsure": "No",
    }
    chronic_map = {"yes": "Yes", "no": "No", "underinvestigation": "No"}
    extra_map = {
        "veryinvolved": "High",
        "somewhatinvolved": "Moderate",
        "minimallyinvolved": "Low",
        "notinvolved": "Low",
        "high": "High",
        "moderate": "Moderate",
        "low": "Low",
    }
    residence_map = {
        "home": "With Family",
        "withfamily": "With Family",
        "hostel": "On-Campus",
        "dorm": "On-Campus",
        "oncampus": "On-Campus",
        "apartment": "Off-Campus",
        "offcampus": "Off-Campus",
        "other": "Off-Campus",
    }

    fin_key = _canonical(raw.get("Financial_Stress", ""))
    fin_map = {
        "veryhigh": 9.0,
        "high": 7.0,
        "moderate": 5.0,
        "low": 3.0,
        "none": 1.0,
    }

    normalized = {
        "Age": _as_float(raw.get("Age", 21), 21.0),
        "Course": course_map.get(_canonical(raw.get("Course", "Others")), str(raw.get("Course", "Others"))),
        "Gender": gender_map.get(_canonical(raw.get("Gender", "Female")), "Female"),
        "CGPA": _as_float(raw.get("CGPA", 3.0), 3.0),
        "Sleep_Quality": sleep_map.get(_canonical(raw.get("Sleep_Quality", "Average")), "Average"),
        "Physical_Activity": pa_map.get(_canonical(raw.get("Physical_Activity", "Moderate")), "Moderate"),
        "Diet_Quality": diet_map.get(_canonical(raw.get("Diet_Quality", "Average")), "Average"),
        "Social_Support": social_map.get(_canonical(raw.get("Social_Support", "Moderate")), "Moderate"),
        "Relationship_Status": rel_map.get(_canonical(raw.get("Relationship") or raw.get("Relationship_Status", "Single")), "Single"),
        "Substance_Use": sub_map.get(_canonical(raw.get("Substance_Use", "Never")), "Never"),
        "Counseling_Service_Use": counseling_map.get(_canonical(raw.get("Counseling") or raw.get("Counseling_Service_Use", "Never")), "Never"),
        "Family_History": fh_map.get(_canonical(raw.get("Family_History", "No")), "No"),
        "Chronic_Illness": chronic_map.get(_canonical(raw.get("Chronic_Illness", "No")), "No"),
        "Financial_Stress": fin_map.get(fin_key, _as_float(raw.get("Financial_Stress", 5), 5.0)),
        "Extracurricular_Involvement": extra_map.get(_canonical(raw.get("Extracurricular") or raw.get("Extracurricular_Involvement", "Moderate")), "Moderate"),
        "Semester_Credit_Load": _as_float(raw.get("Semester") or raw.get("Semester_Credit_Load", 18), 18.0),
        "Residence_Type": residence_map.get(_canonical(raw.get("Residence_Type", "On-Campus")), "On-Campus"),
    }
    return normalized


def load_baseline_assets(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_bundle = joblib.load(root / BASELINE_MODEL_PATH)
    with (root / BASELINE_META_PATH).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model_bundle, meta


def load_random_forest_assets(root: Path) -> dict[str, Any]:
    """Load the trained random forest and derive the preprocessing needed for inference."""
    bundle_path = root / RF_BUNDLE_PATH
    summary_path = root / RF_SUMMARY_PATH
    data_path = root / RF_DATASET_PATH

    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        bundle["summary"] = summary
        return bundle

    model = None
    if (root / RF_MODEL_PATH).exists():
        model = joblib.load(root / RF_MODEL_PATH)

    df = pd.read_csv(data_path)
    df = df.copy()

    bins = [0, 1.25, 2.5, 3.75, 5]
    labels = ["low burnout", "mid burnout", "high burnout"]
    if "burnout_raw_score" in df.columns:
        df["burnout_category"] = pd.cut(
            df["burnout_raw_score"],
            bins=3,
            labels=labels,
            include_lowest=True,
        )

    def stochastic_hotdeck(series: pd.Series, seed: int = 42) -> pd.Series:
        np.random.seed(seed)
        observed = series.dropna().values
        if len(observed) == 0:
            return series
        missing = series.isna()
        if missing.any():
            series = series.copy()
            series.loc[missing] = np.random.choice(observed, size=missing.sum(), replace=True)
        return series

    for col in df.columns:
        if col != "burnout_category":
            df[col] = stochastic_hotdeck(df[col])

    feature_cols = [c for c in df.columns if c not in RF_DROP_COLUMNS]
    x = df[feature_cols].copy()

    encoders: dict[str, LabelEncoder] = {}
    for col in x.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        encoder.fit(x[col].astype(str))
        encoders[col] = encoder

    target_encoder = LabelEncoder()
    target_encoder.fit(df["burnout_category"].astype(str))
    class_names = list(target_encoder.classes_)

    if model is None:
        y = target_encoder.transform(df["burnout_category"].astype(str))
        for col in x.select_dtypes(include=["object"]).columns:
            x[col] = encoders[col].transform(x[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="log2",
            min_samples_split=5,
            min_samples_leaf=1,
            max_depth=None,
            random_state=42,
        )
        model.fit(X_train_sm, y_train_sm)

        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "encoders": encoders,
                "default_values": {
                    col: (float(pd.to_numeric(x[col], errors="coerce").median()) if col in x.select_dtypes(include=["number"]).columns else (str(x[col].mode(dropna=True).iloc[0]) if not x[col].mode(dropna=True).empty else "Unknown"))
                    for col in feature_cols
                },
                "class_names": class_names,
            },
            bundle_path,
        )

    default_values: dict[str, Any] = {}
    for col in feature_cols:
        if col in x.select_dtypes(include=["number"]).columns:
            default_values[col] = float(pd.to_numeric(x[col], errors="coerce").median())
        else:
            mode_values = x[col].mode(dropna=True)
            default_values[col] = str(mode_values.iloc[0]) if not mode_values.empty else "Unknown"

    return {
        "model": model,
        "summary": summary,
        "feature_cols": feature_cols,
        "encoders": encoders,
        "default_values": default_values,
        "class_names": class_names,
    }


def predict_baseline(normalized_input: dict[str, Any], model_bundle: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]
    feats = meta["features"]
    enc_map = meta["encoding_map"]
    class_names = meta["class_names"]

    row = {}
    for feature in feats:
        value = normalized_input.get(feature, "Unknown")
        if feature in enc_map:
            row[feature] = enc_map[feature].get(str(value), 1)
        else:
            row[feature] = value

    x = pd.DataFrame([row], columns=feats).astype(float)
    probs = model.predict_proba(scaler.transform(x))[0]
    pred_idx = int(np.argmax(probs))
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    accuracy = float(metrics.get("Accuracy", metrics.get("accuracy", 0.0)))
    return {
        "predicted_class_index": pred_idx,
        "predicted_class": class_names[pred_idx],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "accuracy": accuracy,
        "model_type": meta.get("model_type", "Multinomial Logistic Regression"),
    }


def build_or_load_kmeans_bundle(root: Path) -> dict[str, Any]:
    bundle_path = root / KMEANS_BUNDLE_PATH
    if bundle_path.exists():
        return joblib.load(bundle_path)

    data_path = root / DATASET_PATH
    df = pd.read_csv(data_path)
    y_true = pd.qcut(df["burnout_raw_score"].astype(float), q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

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
    x = df[feature_cols].copy()

    num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]

    for c in num_cols:
        if x[c].isnull().any():
            x[c] = x[c].fillna(x[c].median())
    for c in cat_cols:
        if x[c].isnull().any():
            x[c] = x[c].fillna("Unknown")

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", Pipeline(steps=[("encode", ohe)]), cat_cols),
        ],
        remainder="drop",
    )

    x_proc = preprocessor.fit_transform(x)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(x_proc)

    map_df = pd.DataFrame({"cluster": clusters, "burnout_q": y_true})
    cluster_to_class = map_df.groupby("cluster")["burnout_q"].agg(lambda s: int(s.value_counts().idxmax())).to_dict()

    default_values = {}
    for c in feature_cols:
        if c in num_cols:
            default_values[c] = float(x[c].median())
        else:
            default_values[c] = str(x[c].mode(dropna=True).iloc[0]) if not x[c].mode(dropna=True).empty else "Unknown"

    bundle = {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "cluster_to_burnout_class": cluster_to_class,
        "default_values": default_values,
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    return bundle


def predict_kmeans(normalized_input: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    feature_cols = bundle["feature_cols"]
    defaults = bundle["default_values"]
    row = {f: normalized_input.get(f, defaults.get(f)) for f in feature_cols}
    x_row = pd.DataFrame([row], columns=feature_cols)

    # Keep data types robust before transform.
    for c in bundle["num_cols"]:
        x_row[c] = pd.to_numeric(x_row[c], errors="coerce").fillna(defaults[c])
    for c in bundle["cat_cols"]:
        x_row[c] = x_row[c].astype(str).fillna(str(defaults[c]))

    x_proc = bundle["preprocessor"].transform(x_row)
    cluster_id = int(bundle["kmeans"].predict(x_proc)[0])
    mapped_class = int(bundle["cluster_to_burnout_class"].get(cluster_id, 1))

    quartile_label = {
        0: "Very Low (Q1)",
        1: "Low (Q2)",
        2: "Moderate (Q3)",
        3: "High (Q4)",
    }.get(mapped_class, "Low (Q2)")

    return {
        "cluster": cluster_id,
        "mapped_burnout_class_index": mapped_class,
        "mapped_burnout_class": quartile_label,
    }


def predict_random_forest(normalized_input: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """Predict burnout class with the trained random forest model."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    default_values = bundle["default_values"]
    encoders = bundle["encoders"]
    class_names = bundle["class_names"]
    summary = bundle.get("summary", {})

    row = {feature: normalized_input.get(feature, default_values.get(feature)) for feature in feature_cols}
    x_row = pd.DataFrame([row], columns=feature_cols)

    for col in x_row.columns:
        if col in encoders:
            value = str(x_row.at[0, col])
            if value not in encoders[col].classes_:
                value = str(default_values.get(col, encoders[col].classes_[0] if len(encoders[col].classes_) else "Unknown"))
            x_row.at[0, col] = value
        else:
            x_row[col] = pd.to_numeric(x_row[col], errors="coerce").fillna(default_values.get(col, 0.0))

    encoded_row = x_row.copy()
    for col in encoders:
        encoded_row[col] = encoders[col].transform(encoded_row[col].astype(str))

    probs = model.predict_proba(encoded_row.astype(float))[0]
    pred_idx = int(np.argmax(probs))

    if not class_names:
        class_names = [str(i) for i in range(len(probs))]

    predicted_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    return {
        "predicted_class_index": pred_idx,
        "predicted_class": predicted_label,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(min(len(class_names), len(probs)))},
        "accuracy": float(summary.get("output", {}).get("accuracy", 0.0)),
        "best_parameters": summary.get("best_parameter", {}),
        "model_type": "Random Forest",
    }


def integrated_predict(user_input: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized = normalize_from_app_answers(user_input)

    baseline_bundle, baseline_meta = load_baseline_assets(root)
    baseline_pred = predict_baseline(normalized, baseline_bundle, baseline_meta)

    rf_bundle = load_random_forest_assets(root)
    rf_pred = predict_random_forest(normalized, rf_bundle)

    kmeans_bundle = build_or_load_kmeans_bundle(root)
    unsupervised_pred = predict_kmeans(normalized, kmeans_bundle)

    return {
        "input_raw": user_input,
        "input_normalized": normalized,
        "baseline_multinomial": baseline_pred,
        "random_forest": rf_pred,
        "unsupervised_kmeans": unsupervised_pred,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated baseline + unsupervised inference for one student input.")
    parser.add_argument("--input-json", help="Single input object as JSON string.")
    parser.add_argument("--input-file", help="Path to JSON file with a single input object.")
    parser.add_argument("--output-file", help="Optional path to save integrated output JSON.")
    return parser.parse_args()


def load_user_input(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.input_json) == bool(args.input_file):
        raise ValueError("Provide exactly one of --input-json or --input-file.")

    if args.input_json:
        return json.loads(args.input_json)

    input_path = Path(args.input_file)
    with input_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Input JSON list is empty.")
        if not isinstance(payload[0], dict):
            raise ValueError("Input JSON list must contain objects.")
        return payload[0]

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    return payload


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    user_input = load_user_input(args)
    result = integrated_predict(user_input, root)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
