"""
Model training utilities for student mental health stress prediction.

This module provides functions to train and evaluate multiple ML models:
  - Logistic Regression: Multi-class stress level classification
  - Gradient Boosting: Ensemble-based stress prediction with better generalization
  - KMeans: Unsupervised clustering for behavioral pattern discovery
  
Also includes preprocessing pipelines (imputation, scaling, encoding) and feature engineering
for the mental wellbeing target variable.

Key Design:
  - Logistic Regression selected for interpretability and fast inference
  - Gradient Boosting chosen for robustness and handling non-linear relationships
  - KMeans for unsupervised segmentation into 3 behavioral groups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, roc_auc_score, roc_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_PATH = "Dataset/students_mental_health_survey.csv"


def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    """
    Validate that required columns exist in DataFrame, raise error if missing.
    
    Raises:
        ValueError: If any required column is missing from the DataFrame.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def _safe_mode(series: pd.Series) -> Optional[Any]:
    """
    Return most frequent value in series, or None if empty.
    Used for handling missing values in categorical imputation.
    """
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    """
    Extract numeric column from DataFrame, fill NaNs with default value.
    
    Args:
        df: Input DataFrame
        column: Column name to extract
        default: Default value for missing entries
    
    Returns:
        pd.Series: Numeric series with defaults filled
    """
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _mapped_or_default(df: pd.DataFrame, column: str, mapping: Dict[str, float], default: float) -> pd.Series:
    """
    Map categorical values using provided mapping, fill unmapped values with default.
    
    Args:
        df: Input DataFrame
        column: Column name containing categorical values
        mapping: Dict mapping categorical values to numeric scores
        default: Default value for missing/unmapped entries
    
    Returns:
        pd.Series: Mapped numeric series
    """
    if column in df.columns:
        return df[column].map(mapping).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)



def load_dataset() -> pd.DataFrame:
    """
    Load student mental health dataset from CSV.
    
    Returns:
        pd.DataFrame: Raw dataset with empty strings replaced by NaN
    """
    df = pd.read_csv(DATA_PATH)
    df = df.replace("", np.nan)
    return df


def build_wellbeing_target(df: pd.DataFrame) -> pd.Series:
    """
    Engineer composite mental wellbeing percentage target variable (0-100).
    
    Combines positive factors (sleep, activity, diet, social support, extracurriculars)
    and subtracts negative stress/depression/anxiety impacts.
    
    Formula:
        wellbeing% = 100 - stress_impact(40%) - depression_impact(25%) - anxiety_impact(20%)
                     + lifestyle_bonuses(sleep, activity, diet, support, activities)
    
    Args:
        df: Input DataFrame with required columns
    
    Returns:
        pd.Series: Mental wellbeing percentage clipped to [0, 100]
    
    Design Rationale:
        - Stress (40% weight): Strongest driver of overall wellbeing decline
        - Depression (25% weight): Significant impact on mental health
        - Anxiety (20% weight): Additional stress component
        - Lifestyle bonuses reward positive habits (sleep, exercise, diet, social support)
    """
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
    """
    Build preprocessing pipeline for both numeric and categorical features.
    
    Pipeline:
        - Numeric: Impute via median, then StandardScale
        - Categorical: Impute via most_frequent mode, then OneHotEncode
    
    Args:
        x: Feature DataFrame with mixed numeric and categorical columns
    
    Returns:
        ColumnTransformer: Scikit-learn pipeline for preprocessing
    
    Design Rationale:
        - Median imputation: Robust to outliers in numeric features
        - StandardScaler: Required for Logistic Regression and KMeans
        - OneHotEncoder: Converts categorical features to sparse binary columns
    """
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




def train_logistic_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train Logistic Regression model for stress level classification.
    
    **Model Selection Rationale:**
    - Interpretable: Provides coefficient-based feature importance
    - Fast to train and predict: Suitable for real-time web app inference
    - Probabilistic: Can output confidence scores for predictions
    - Well-suited for multi-class (stress: 1-5) classification
    
    Args:
        df: Dataset containing 'Stress_Level' target and feature columns
    
    Returns:
        Dict with comprehensive metrics:
            - 'model': Trained Pipeline with preprocessing + LogisticRegression
            - 'accuracy': Test set accuracy (0-1)
            - 'f1_score': Weighted F1 score
            - 'precision': Weighted precision score
            - 'recall': Weighted recall score
            - 'confusion_matrix': Confusion matrix array
            - 'y_test': Test labels for advanced analysis
            - 'y_pred': Model predictions on test set
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'feature_cols': List of feature column names
            - 'test_size': Number of test samples
    
    Raises:
        ValueError: If Stress_Level column missing or no usable features found
    """
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
    
    # Comprehensive metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
        "y_test": y_test.values,
        "y_pred": preds,
        "mse": mse,
        "mae": mae,
        "feature_cols": model_features,
        "test_size": len(y_test),
    }


def train_gradient_boosting(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train Gradient Boosting model for stress level classification.
    
    **Model Selection Rationale:**
    - Captures non-linear relationships between features and stress
    - Ensemble method reduces overfitting through sequential error correction
    - Handles mixed feature types well
    - Generally outperforms Logistic Regression on complex datasets
    
    Args:
        df: Dataset containing 'Stress_Level' target and feature columns
    
    Returns:
        Dict with comprehensive metrics:
            - 'model': Trained Pipeline with preprocessing + GradientBoostingClassifier
            - 'accuracy': Test set accuracy (0-1)
            - 'f1_score': Weighted F1 score
            - 'precision': Weighted precision score
            - 'recall': Weighted recall score
            - 'confusion_matrix': Confusion matrix array
            - 'y_test': Test labels
            - 'y_pred': Model predictions on test set
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'feature_cols': List of feature column names
            - 'test_size': Number of test samples
    
    Raises:
        ValueError: If Stress_Level column missing or no usable features found
    """
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
    
    # Comprehensive metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
        "y_test": y_test.values,
        "y_pred": preds,
        "mse": mse,
        "mae": mae,
        "feature_cols": model_features,
        "test_size": len(y_test),
    }


def train_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Train KMeans clustering model for behavioral pattern discovery.
    
    **Model Selection Rationale:**
    - Unsupervised: Discovers naturally-occurring student groups without labels
    - 3 clusters: Stable pattern, Moderate strain, High pressure
    - Complements supervised stress predictions with behavioral segmentation
    - Results drive personalized behavior remarks (e.g., "improve recovery habits")
    
    Args:
        df: Dataset with feature columns
        n_clusters: Number of behavioral clusters (default: 3)
    
    Returns:
        Dict with comprehensive clustering metrics:
            - 'model': Trained KMeans instance
            - 'labels': Cluster assignment array for training data
            - 'silhouette_score': Silhouette coefficient (-1 to 1, higher is better)
            - 'inertia': Sum of squared distances to nearest cluster center
            - 'davies_bouldin_index': Cluster separation metric (lower is better)
            - 'feature_cols': List of feature column names
            - 'n_clusters': Number of clusters
    
    Raises:
        ValueError: If no usable features found
    """
    model_features = [col for col in df.columns if col not in ["Stress_Level", "Depression_Score", "Anxiety_Score"]]
    if not model_features:
        raise ValueError("No usable feature columns found for kmeans.")
    x = df[model_features].copy()
    preprocess = build_preprocessor(x)
    x_proc = preprocess.fit_transform(x)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x_proc)
    labels = kmeans.labels_
    
    # Clustering quality metrics
    silhouette = silhouette_score(x_proc, labels)
    davies_bouldin = davies_bouldin_score(x_proc, labels)
    inertia = kmeans.inertia_
    
    return {
        "model": kmeans,
        "labels": labels,
        "silhouette_score": silhouette,
        "inertia": inertia,
        "davies_bouldin_index": davies_bouldin,
        "feature_cols": model_features,
        "n_clusters": n_clusters,
    }


def build_input_row(feature_cols: List[str], user_answers: Dict[str, Any], reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature DataFrame from user questionnaire answers.
    
    For missing answers, imputes with reference dataset statistics:
    - Numeric columns: Median value
    - Categorical columns: Most frequent value (mode)
    
    This ensures models always receive complete feature vectors.
    
    Args:
        feature_cols: List of required feature column names
        user_answers: Dict of user questionnaire responses (may be partial)
        reference_df: Full dataset used for computing defaults
    
    Returns:
        pd.DataFrame: Single-row DataFrame with complete features ready for model prediction
    
    Example:
        >>> user_input = {'Age': 22, 'Sleep_Quality': 'Good'}
        >>> prediction_row = build_input_row(model.feature_cols, user_input, training_df)
        >>> stress_pred = model.predict(prediction_row)
    """
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
