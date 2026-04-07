"""
Self-contained training script for the Gaming and Mental Health regression model.
Downloads data from Kaggle, trains models, and saves the best one with scaler and metadata.
"""

import json
import os
import pickle

import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_NAME = "sharmajicoder/gaming-and-mental-health"
TARGET_COLUMN = "addiction_level"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def download_data(dataset_name: str) -> pd.DataFrame:
    """Download dataset from Kaggle and return as DataFrame."""
    download_path = kagglehub.dataset_download(dataset_name)
    for root, _, files in os.walk(download_path):
        for f in files:
            if f.endswith(".csv"):
                return pd.read_csv(os.path.join(root, f))
    raise FileNotFoundError(f"No CSV file found in {download_path}")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: downcast types."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def train_and_select_best(X_train, X_val, y_train, y_val):
    """Train multiple models, return best model and fitted scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    best_model = None
    best_score = -np.inf
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = r2_score(y_val, model.predict(X_val_scaled))
        print(f"{name}: R2 = {score:.4f}")
        if score > best_score:
            best_model = model
            best_score = score
            best_name = name

    print(f"\nBest model: {best_name} (R2 = {best_score:.4f})")
    return best_model, scaler, best_name


def main():
    print("Downloading dataset...")
    df = download_data(DATASET_NAME)
    df = preprocess(df)

    X = df.select_dtypes(include=np.number).drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models...")
    model, scaler, model_type = train_and_select_best(X_train, X_val, y_train, y_val)

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "model_type": model_type,
        "target": TARGET_COLUMN,
        "features": feature_names,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
