"""
Тестирование обученных моделей на тестовых данных.
Загрузка моделей через pickle и предсказания на test_preprocessed.csv.
"""

import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_model(model_path: str):
    """Загрузка модели из pickle файла."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_test_data(test_path: str) -> pd.DataFrame:
    """Загрузка тестовых данных."""
    test_df = pd.read_csv(test_path)
    return test_df


def prepare_features(test_df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка признаков из тестовых данных."""
    feature_cols = [
        'Age', 'Gender', 'Occupation', 'Device_Type',
        'Daily_Phone_Hours', 'Social_Media_Hours', 'Sleep_Hours',
        'App_Usage_Count', 'Caffeine_Intake_Cups',
        'Weekend_Screen_Time_Hours'
    ]
    return test_df[feature_cols]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    """Оценка модели и вывод метрик."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')


    return {'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m}


def main():
    """Основная функция для тестирования моделей."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    # Пути к моделям
    prod_model_path = os.path.join(data_dir, 'models', 'p_model.pkl')
    stress_model_path = os.path.join(data_dir, 'models', 's_model.pkl')

    # Путь к тестовым данным
    test_path = os.path.join(data_dir, 'test', 'test_preprocessed.csv')

    # Загрузка тестовых данных
    test_df = load_test_data(test_path)

    # Подготовка признаков
    X_test = prepare_features(test_df)

    # Истинные метки
    y_test_prod = test_df['Productivity_Class']
    y_test_stress = test_df['Stress_Class']

    # Загрузка и тестирование модели продуктивности
    if os.path.exists(prod_model_path):
        prod_model = load_model(prod_model_path)
        prod_metrics = evaluate_model(prod_model, X_test, y_test_prod, "Productivity Model")
    else:
        print(f"⚠️ Model not found: {prod_model_path}")

    # Загрузка и тестирование модели стресса

    if os.path.exists(stress_model_path):
        stress_model = load_model(stress_model_path)
        stress_metrics = evaluate_model(stress_model, X_test, y_test_stress, "Stress Model")
    else:
        print(f"⚠️ Model not found: {stress_model_path}")


    if 'prod_metrics' in dir():
        print(
            f"Model 'Productivity' is {prod_metrics['accuracy']:.4f} accuracy.",end='\t\t')
    if 'stress_metrics' in dir():
        print(
            f"Model 'Stress' is {stress_metrics['accuracy']:.4f} accuracy.",end='')


if __name__ == "__main__":
    main()
