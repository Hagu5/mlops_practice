"""
Скрипт для скачивания набора данных с Kaggle и разделения на обучающую и тестовую выборки.
"""

import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def download_dataset_from_kaggle(dataset_name: str, download_path: str) -> str:
    """
    Скачивает набор данных с Kaggle с использованием API.
    
    Args:
        dataset_name: Имя датасета в формате 'owner/dataset-name'
        download_path: Путь для сохранения скачанных файлов
    
    Returns:
        Путь к директории с данными
    """
    os.makedirs(download_path, exist_ok=True)

    # Скачивание датасета через kaggle API
    kagglehub.dataset_download(dataset_name, output_dir=download_path, force_download=True)

    return download_path


def find_csv_file(directory: str) -> str:
    """
    Находит CSV файл в указанной директории.
    
    Args:
        directory: Путь к директории
    
    Returns:
        Путь к найденному CSV файлу
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)
    raise FileNotFoundError(f"CSV файл не найден в директории {directory}")


def split_and_save_data(
        data_path: str,
        train_path: str,
        test_path: str,
        test_size: float = 0.2,
        random_state: int = 42
) -> None:
    """
    Загружает данные, разделяет на обучающую и тестовую выборки и сохраняет их.
    
    Args:
        data_path: Путь к CSV файлу с данными
        train_path: Путь для сохранения обучающей выборки
        test_path: Путь для сохранения тестовой выборки
        test_size: Доля тестовой выборки
        random_state: Seed для воспроизводимости
    """
    # Загрузка данных
    df = pd.read_csv(data_path)


    # Разделение на обучающую и тестовую выборки
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    # Сохранение выборок
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)



def main():
    """Основная функция для скачивания и обработки данных."""
    # Параметры
    dataset_name = "amar5693/screen-time-sleep-and-stress-analysis-dataset"
    download_dir = "data/raw"
    train_path = "data/train/train.csv"
    test_path = "data/test/test.csv"
    test_size = 0.2


    # Скачивание датасета
    download_dataset_from_kaggle(dataset_name, download_dir)

    # Поиск CSV файла
    csv_path = find_csv_file(download_dir)

    # Создание директорий для train/test
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # Разделение и сохранение данных
    split_and_save_data(csv_path, train_path, test_path, test_size)



if __name__ == "__main__":
    main()
