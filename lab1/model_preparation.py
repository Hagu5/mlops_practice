"""
Подготовка данных для обучения моделей.
Загрузка предобработанных train данных и разделение на train/val/test.
"""
import os
import pickle
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=UserWarning)


def load_preprocessed_data(train_path: str) -> pd.DataFrame:
    """Загрузка предобработанных данных."""
    train_df = pd.read_csv(train_path)
    return train_df


def prepare_features_and_targets(train_df: pd.DataFrame) -> tuple:
    """
    Подготовка признаков и целевых переменных.
    """
    X = train_df.select_dtypes(include=np.number).drop(columns=['addiction_level'])
    y = train_df['addiction_level']
    return X, y


def split_data(X, y, val_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Разделение данных на train/val.
    
    Сначала делим на train и val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


def regression_model(X_train, X_val, y_train, y_val) -> tuple:
    models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor()
    }
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    best_model_score = 0
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        if (best_model is None):
            best_model = model
            best_model_score = r2_score(y_val, preds)
        elif (best_model_score < r2_score(y_val, preds)):
            best_model = model
            best_model_score = r2_score(y_val, preds)
        print(name, "R2:", r2_score(y_val, preds))
    return best_model


def main():
    """Основная функция для подготовки данных."""
    # Путь к предобработанным данным
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    train_path = os.path.join(data_dir, 'train', 'train_preprocessed.csv')
    models_path = os.path.join(data_dir, 'models')
    model_path = os.path.join(models_path, 'model.pkl')

    train_df = load_preprocessed_data(train_path)

    X, y = prepare_features_and_targets(train_df)
    X_train, X_val, y_train, y_val = split_data(X, y)
    model = regression_model(X_train, X_val, y_train, y_val)
    os.makedirs(models_path, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
