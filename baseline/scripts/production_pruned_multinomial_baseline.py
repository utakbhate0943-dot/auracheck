"""
Production baseline script: 4-class balanced multinomial model (VIF-pruned features).

What it does:
1) Trains the agreed baseline model.
2) Saves model artifacts (scaler + classifier + metadata).
3) Exposes a simple prediction function for one student profile.
4) Writes one example prediction output.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score

ROOT = "/Users/edgarvidriales/Desktop/AuraCheck/auracheck"
DATA_PATH = "/Users/edgarvidriales/Desktop/AuraCheck/auracheck/Dataset/students_mental_health_survey_with_burnout_final.csv"
OUT_DIR = os.path.join(ROOT, "baseline", "outputs", "final_baseline_model")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "production_pruned_multinomial_model.joblib")
META_PATH = os.path.join(OUT_DIR, "production_pruned_multinomial_metadata.json")
METRICS_PATH = os.path.join(OUT_DIR, "production_pruned_multinomial_metrics.csv")
EXAMPLE_OUT = os.path.join(OUT_DIR, "production_pruned_multinomial_example_prediction.json")

FEATURES_ALL = [
    "Course", "Age", "Gender", "CGPA", "Sleep_Quality", "Physical_Activity",
    "Diet_Quality", "Social_Support", "Relationship_Status", "Substance_Use",
    "Counseling_Service_Use", "Family_History", "Chronic_Illness",
    "Financial_Stress", "Extracurricular_Involvement", "Semester_Credit_Load",
    "Residence_Type"
]

# Agreed final pruned features (remove CGPA, Age, Semester_Credit_Load)
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

CLASS_NAMES = ["Very Low (Q1)", "Low (Q2)", "Moderate (Q3)", "High (Q4)"]


def preprocess(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in X.columns:
        if X[c].isnull().any():
            if X[c].dtype == "object":
                X[c] = X[c].fillna("Unknown")
            else:
                X[c] = X[c].fillna(X[c].median())

    for c, mp in ENCODING_MAP.items():
        if c in X.columns:
            X[c] = X[c].astype(str).map(mp).fillna(1)

    return X.astype(float)


def train_and_save():
    df = pd.read_csv(DATA_PATH)
    X = preprocess(df, FEATURES_PRUNED)

    y, bins = pd.qcut(
        df["burnout_raw_score"].astype(float),
        q=4,
        labels=[0, 1, 2, 3],
        duplicates="drop",
        retbins=True,
    )
    y = y.astype(int)

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
    rec = recall_score(y_test, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Kappa": float(cohen_kappa_score(y_test, pred)),
        "Recall_VeryLow": float(rec[0]),
        "Recall_Low": float(rec[1]),
        "Recall_Moderate": float(rec[2]),
        "Recall_High": float(rec[3]),
        "Macro_Recall": float(rec.mean()),
    }

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    joblib.dump({"scaler": scaler, "model": clf}, MODEL_PATH)

    meta = {
        "model_type": "Balanced Multinomial Logistic Regression",
        "features": FEATURES_PRUNED,
        "class_names": CLASS_NAMES,
        "quartile_bins": [float(x) for x in bins],
        "encoding_map": ENCODING_MAP,
        "metrics": metrics,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Saved: {MODEL_PATH}")
    print(f"✓ Saved: {META_PATH}")
    print(f"✓ Saved: {METRICS_PATH}")


def predict_student(student_answers: dict):
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    clf = bundle["model"]

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feats = meta["features"]
    class_names = meta["class_names"]

    row = {}
    for f_name in feats:
        val = student_answers.get(f_name, "Unknown")
        if f_name in ENCODING_MAP:
            row[f_name] = ENCODING_MAP[f_name].get(str(val), 1)
        else:
            row[f_name] = val

    X_row = pd.DataFrame([row], columns=feats).astype(float)
    X_row_s = scaler.transform(X_row)
    probs = clf.predict_proba(X_row_s)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class_index": pred_idx,
        "predicted_class": class_names[pred_idx],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def main():
    train_and_save()

    # Example inference call (same schema your report uses)
    example_student = {
        "Course": "Engineering",
        "Gender": "Female",
        "Sleep_Quality": "Average",
        "Physical_Activity": "Low",
        "Diet_Quality": "Average",
        "Social_Support": "Low",
        "Relationship_Status": "Single",
        "Substance_Use": "Occasionally",
        "Counseling_Service_Use": "Never",
        "Family_History": "Yes",
        "Chronic_Illness": "No",
        "Financial_Stress": 8,
        "Extracurricular_Involvement": "Low",
        "Residence_Type": "On-Campus",
    }

    pred = predict_student(example_student)
    with open(EXAMPLE_OUT, "w", encoding="utf-8") as f:
        json.dump({"input": example_student, "prediction": pred}, f, indent=2)

    print(f"✓ Saved: {EXAMPLE_OUT}")


if __name__ == "__main__":
    main()
