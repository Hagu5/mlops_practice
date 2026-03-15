"""
Предобработка данных для задачи классификации продуктивности и уровня стресса.
"""

import os

import pandas as pd




def load_data(train_path: str, test_path: str) -> tuple:
    """Загрузка train и test данных."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Полная предобработка данных.
    """
    dfs = [train_df,test_df]
    for df in dfs:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
        
        df['addiction_category'] = pd.cut(df['addiction_level'], 
                                  bins=3, labels=['Low','Medium','High'])
        


    return dfs[train_df], dfs[test_df]


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

    train_df, test_df = preprocess_data(train_df, test_df)

    save_data(train_df, test_df, train_output_path, test_output_path)


if __name__ == "__main__":
    main()
