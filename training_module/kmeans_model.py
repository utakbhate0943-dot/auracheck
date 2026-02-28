import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_kmeans_from_csv(csv_path: str, n_clusters: int = 3):
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
