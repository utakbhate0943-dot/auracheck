"""
KMeans clustering model for behavioral student grouping from CSV.

Implements unsupervised learning to discover natural student groups with different
stress management patterns. Used to provide tailored behavioral recommendations.

Functions:
    train_kmeans_from_csv: Train KMeans clustering model from CSV file
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def train_kmeans_from_csv(csv_path: str, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Train KMeans clustering model from CSV dataset file.
    
    **Model Design:**
    - Unsupervised clustering: discovers natural student groups without labels
    - n_clusters=3: Typically identifies:
        * Cluster 0: "Stable pattern" - good stress management habits
        * Cluster 1: "Moderate strain" - some stress but manageable
        * Cluster 2: "High pressure" - significant stress needing intervention
    - Used to generate behavior-based remarks (e.g., "improve recovery habits")
    - Complements supervised stress prediction models
    
    **Preprocessing:**
    - Numeric: Median imputation + StandardScaling
    - Categorical: Mode imputation + OneHotEncoding
    - Feature scaling essential for K-Means distance calculation
    
    Args:
        csv_path: Path to CSV file containing dataset
        n_clusters: Number of behavioral clusters to discover (default: 3)
    
    Returns:
        Dict with keys:
            - 'model': Trained KMeans instance
            - 'labels': Cluster assignment array for training data samples
            - 'feature_cols': List of feature column names used
    
    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If no usable features found in dataset
    
    Example:
        >>> result = train_kmeans_from_csv('data/students_mental_health.csv', n_clusters=3)
        >>> user_cluster = result['model'].predict(new_data_preprocessed)
        >>> print(f"User assigned to cluster {user_cluster[0]}")
    """
    df = pd.read_csv(csv_path)
    model_features = [col for col in df.columns if col not in ["Stress_Level", "Depression_Score", "Anxiety_Score"]]
    x = df[model_features].copy()
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
    x_proc = preprocess.fit_transform(x)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x_proc)
    labels = kmeans.labels_
    return {"model": kmeans, "labels": labels, "feature_cols": model_features}
