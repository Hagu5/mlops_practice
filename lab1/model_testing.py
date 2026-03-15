"""
Тестирование обученных моделей на тестовых данных.
Загрузка моделей через pickle и предсказания на test_preprocessed.csv.
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def load_model(model_path: str):
    """Загрузка модели из pickle файла."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_test_data(test_path: str) -> tuple:
    """Загрузка тестовых данных."""
    test_df = pd.read_csv(test_path)
    X = test_df.select_dtypes(include=np.number).drop(columns=['addiction_level'])
    y = test_df['addiction_level']
    return X, y



def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Оценка модели и вывод метрик."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    return {'R2': r2, 'MAE': mae, 'MSE': mse}


def main():
    """Основная функция для тестирования моделей."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    model_path = os.path.join(data_dir, 'models', 'model.pkl')

    # Путь к тестовым данным
    test_path = os.path.join(data_dir, 'test', 'test_preprocessed.csv')

    # Загрузка тестовых данных
    X_test, y_test = load_test_data(test_path)

    # Загрузка и тестирование модели продуктивности
    if os.path.exists(model_path):
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
    else:
        print(f"⚠️ Model not found: {model_path}")
        exit()
    
    print(
        f"Model is {metrics['R2']:.4f} R2.",end='')


if __name__ == "__main__":
    main()
