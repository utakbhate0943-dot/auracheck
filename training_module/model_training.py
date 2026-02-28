import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "Dataset/students_mental_health_survey.csv"


def _require_columns(df: pd.DataFrame, required: list[str], context: str):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _mapped_or_default(df: pd.DataFrame, column: str, mapping: dict, default: float) -> pd.Series:
    if column in df.columns:
        return df[column].map(mapping).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.replace("", np.nan)
    return df


def build_wellbeing_target(df: pd.DataFrame) -> pd.Series:
    sleep_bonus = _mapped_or_default(df, "Sleep_Quality", {"Poor": 0, "Average": 5, "Good": 10}, 5)
    activity_bonus = _mapped_or_default(df, "Physical_Activity", {"Low": 0, "Moderate": 5, "High": 10}, 5)
    diet_bonus = _mapped_or_default(df, "Diet_Quality", {"Poor": 0, "Average": 4, "Good": 8}, 4)
    support_bonus = _mapped_or_default(df, "Social_Support", {"Low": 0, "Moderate": 4, "High": 8}, 4)
    extra_bonus = _mapped_or_default(df, "Extracurricular_Involvement", {"Low": 0, "Moderate": 3, "High": 6}, 3)

    stress_impact = (_series_or_default(df, "Stress_Level", 2.5) / 5.0) * 40
    depression_impact = (_series_or_default(df, "Depression_Score", 2.5) / 5.0) * 25
    anxiety_impact = (_series_or_default(df, "Anxiety_Score", 2.5) / 5.0) * 20

    wellbeing = 100 - stress_impact - depression_impact - anxiety_impact + sleep_bonus + activity_bonus + diet_bonus + support_bonus + extra_bonus
    return np.clip(wellbeing, 0, 100)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])



# Logistic Regression Model Training
def train_logistic_regression(df: pd.DataFrame):
    _require_columns(df, ["Stress_Level"], "logistic regression")
    model_features = [col for col in df.columns if col not in ["Stress_Level", "Depression_Score", "Anxiety_Score"]]
    if not model_features:
        raise ValueError("No usable feature columns found for logistic regression.")
    x = df[model_features].copy()
    y = df["Stress_Level"].astype(float).round().astype(int)
    preprocess = build_preprocessor(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2500)),
    ])
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "feature_cols": model_features,
    }

# Gradient Boosting Model Training
def train_gradient_boosting(df: pd.DataFrame):
    _require_columns(df, ["Stress_Level"], "gradient boosting")
    model_features = [col for col in df.columns if col not in ["Stress_Level", "Depression_Score", "Anxiety_Score"]]
    if not model_features:
        raise ValueError("No usable feature columns found for gradient boosting.")
    x = df[model_features].copy()
    y = df["Stress_Level"].astype(float).round().astype(int)
    preprocess = build_preprocessor(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "feature_cols": model_features,
    }

# KMeans Unsupervised Model Training
def train_kmeans(df: pd.DataFrame, n_clusters: int = 3):
    model_features = [col for col in df.columns if col not in ["Stress_Level", "Depression_Score", "Anxiety_Score"]]
    if not model_features:
        raise ValueError("No usable feature columns found for kmeans.")
    x = df[model_features].copy()
    preprocess = build_preprocessor(x)
    x_proc = preprocess.fit_transform(x)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x_proc)
    labels = kmeans.labels_
    return {
        "model": kmeans,
        "labels": labels,
        "feature_cols": model_features,
    }


def build_input_row(feature_cols: list[str], user_answers: dict, reference_df: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for col in feature_cols:
        if col in user_answers and user_answers[col] not in [None, ""]:
            row[col] = user_answers[col]
        else:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                row[col] = float(reference_df[col].median())
            else:
                row[col] = _safe_mode(reference_df[col])
    return pd.DataFrame([row])
