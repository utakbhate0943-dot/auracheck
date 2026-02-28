import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def _require_columns(df: pd.DataFrame, required: list[str], context: str):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")

def train_logistic_regression_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    _require_columns(df, ["Stress_Level"], "logistic regression")
    model_features = [col for col in df.columns if col not in ["Stress_Level", "Depression_Score", "Anxiety_Score"]]
    if not model_features:
        raise ValueError("No usable feature columns found for logistic regression.")
    x = df[model_features].copy()
    y = df["Stress_Level"].astype(float).round().astype(int)
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), x.select_dtypes(include=[np.number]).columns.tolist()),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), [col for col in x.columns if col not in x.select_dtypes(include=[np.number]).columns.tolist()]),
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2500)),
    ])
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return {"model": model, "accuracy": acc, "f1_score": f1, "feature_cols": model_features}
