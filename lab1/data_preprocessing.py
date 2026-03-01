"""
Предобработка данных для задачи классификации продуктивности и уровня стресса.
"""

import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(train_path: str, test_path: str) -> tuple:
    """Загрузка train и test данных."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def categorize_score(score: int, low_max: int = 4, high_min: int = 7) -> str:
    """
    Разбиение числовой оценки на 3 класса.
    
    Args:
        score: числовая оценка (1-10)
        low_max: максимальное значение для класса 'low'
        high_min: минимальное значение для класса 'high'
    
    Returns:
        'low', 'medium' или 'high'
    """
    if score <= low_max:
        return 'low'
    elif score >= high_min:
        return 'high'
    else:
        return 'medium'


def encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       categorical_cols: list) -> tuple:
    """
    Label encoding для категориальных признаков.
    Обучение encoder на train данных, применение к train и test.
    """
    encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        # Объединяем train и test для получения всех возможных категорий
        all_values = pd.concat([train_df[col], test_df[col]]).unique()
        encoder.fit(all_values)

        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

        encoders[col] = encoder

    return train_df, test_df, encoders


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Полная предобработка данных.
    
    1. Категоризация Work_Productivity_Score и Stress_Level
    2. Label Encoding для категориальных признаков
    """
    # Категориальные признаки для encoding
    categorical_cols = ['Gender', 'Occupation', 'Device_Type', "Productivity_Class", "Stress_Class"]

    # Категоризация целевых переменных
    train_df['Productivity_Class'] = train_df['Work_Productivity_Score'].apply(
        lambda x: categorize_score(x)
    )
    train_df['Stress_Class'] = train_df['Stress_Level'].apply(
        lambda x: categorize_score(x)
    )

    test_df['Productivity_Class'] = test_df['Work_Productivity_Score'].apply(
        lambda x: categorize_score(x)
    )
    test_df['Stress_Class'] = test_df['Stress_Level'].apply(
        lambda x: categorize_score(x)
    )

    # Label Encoding для категориальных признаков
    train_df, test_df, encoders = encode_categorical(
        train_df, test_df, categorical_cols
    )

    return train_df, test_df, encoders


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
              train_output_path: str, test_output_path: str) -> None:
    """
    Сохранение предобработанных данных рядом с оригинальными файлами.
    """
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)



def main():
    """Основная функция."""
    # Пути к данным
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    train_path = os.path.join(data_dir, 'train', 'train.csv')
    test_path = os.path.join(data_dir, 'test', 'test.csv')

    # Пути для сохранения рядом с оригинальными файлами
    train_output_path = os.path.join(data_dir, 'train', 'train_preprocessed.csv')
    test_output_path = os.path.join(data_dir, 'test', 'test_preprocessed.csv')

    train_df, test_df = load_data(train_path, test_path)

    train_df, test_df, encoders = preprocess_data(train_df, test_df)

    save_data(train_df, test_df, train_output_path, test_output_path)


if __name__ == "__main__":
    main()
