"""
Подготовка данных для обучения моделей.
Загрузка предобработанных train данных и разделение на train/val/test.
"""
import os
import pickle
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


def load_preprocessed_data(train_path: str) -> pd.DataFrame:
    """Загрузка предобработанных данных."""
    train_df = pd.read_csv(train_path)
    return train_df


def prepare_features_and_targets(train_df: pd.DataFrame) -> tuple:
    """
    Подготовка признаков и целевых переменных.
    
    Returns:
        X, y_prod, y_stress
    """
    # Признаки (исключаем User_ID и целевые переменные)
    feature_cols = [
        'Age', 'Gender', 'Occupation', 'Device_Type',
        'Daily_Phone_Hours', 'Social_Media_Hours', 'Sleep_Hours',
        'App_Usage_Count', 'Caffeine_Intake_Cups',
        'Weekend_Screen_Time_Hours'
    ]

    X = train_df[feature_cols]

    # Целевые переменные
    y_prod = train_df['Productivity_Class']
    y_stress = train_df['Stress_Class']

    return X, y_prod, y_stress


def split_data(X, y, test_size: float = 0.2, val_size: float = 0.25,
               random_state: int = 42) -> tuple:
    """
    Разделение данных на train/val/test.
    
    Сначала делим на train+val и test, затем train+val на train и val.
    """
    # Сначала: train+val vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# Сетки гиперпараметров для каждой модели
HYPERPARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20, ],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt']
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}


def get_base_model(model_name: str):
    """Возвращает базовую модель без параметров."""
    if model_name == 'Random Forest':
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        return XGBClassifier(random_state=42, use_label_encoder=False,
                             eval_metric='mlogloss', verbosity=0)
    else:
        return LGBMClassifier(random_state=42, verbose=-1)


def tuning_model(X_train, X_test, y_train, y_test, best_prod_name):
    """
    Автоматический подбор гиперпараметров с помощью GridSearchCV.
    """
    print(f'🔧 Auto-tuning: {best_prod_name}')
    print('─' * 50)

    # Получаем базовую модель и сетку параметров
    base_model = get_base_model(best_prod_name)
    param_grid = HYPERPARAM_GRIDS[best_prod_name]

    # GridSearchCV для поиска лучших параметров
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        verbose=3
    )

    print(f'   Searching best parameters...')
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f'\n   ✅ Best parameters found:')
    for param, value in best_params.items():
        print(f'      {param}: {value}')

    # Предсказания на тесте
    y_pred_prod_tuned = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_prod_tuned)
    f1_w = f1_score(y_test, y_pred_prod_tuned, average='weighted')
    f1_m = f1_score(y_test, y_pred_prod_tuned, average='macro')

    print(f'\n✅ Tuned {best_prod_name} results:')
    print(f'   Test Accuracy:    {acc:.4f}')
    print(f'   Test F1 Weighted: {f1_w:.4f}')
    print(f'   Test F1 Macro:    {f1_m:.4f}')

    return best_model, acc, f1_w, f1_m


def classification_model(X_train, X_test, y_train, y_test, is_tuned=True) -> tuple:
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False,
                                 eval_metric='mlogloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    }
    models_tuned = {
        'Random Forest': RandomForestClassifier(max_depth=20,
                                                max_features="sqrt",
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                n_estimators=300, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False,
                                 eval_metric='mlogloss', verbosity=0),
        'LightGBM': LGBMClassifier(colsample_bytree=0.8,
                                   learning_rate=0.1,
                                   max_depth=5,
                                   n_estimators=300,
                                   num_leaves=31,
                                   subsample=0.6, random_state=42, verbose=-1)
    }
    results = {}

    for name, model in models_tuned.items() if is_tuned else models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        f1_m = f1_score(y_test, y_pred, average='macro')

        results[name] = {'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m, 'model': model}

    best_prod_name = max(results, key=lambda k: results[k]['f1_weighted'])
    if not is_tuned:
        best_model, acc, f1_w, f1_m = tuning_model(X_train, X_test, y_train, y_test, best_prod_name)
    else:
        best_model = results[best_prod_name].get('model')
    return best_model


def main():
    """Основная функция для подготовки данных."""
    # Путь к предобработанным данным
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    train_path = os.path.join(data_dir, 'train', 'train_preprocessed.csv')
    models_path = os.path.join(data_dir, 'models')
    p_model_path = os.path.join(models_path, 'p_model.pkl')
    s_model_path = os.path.join(models_path, 's_model.pkl')

    train_df = load_preprocessed_data(train_path)

    X, y_prod, y_stress = prepare_features_and_targets(train_df)

    # Split — Productivity target
    X_train_p, X_test_p, y_train_p, y_test_p = split_data(
        X, y_prod, test_size=0.2, val_size=0.25, random_state=42
    )

    # Split — Stress target (same indices for fair comparison)
    X_train_s, X_test_s, y_train_s, y_test_s = split_data(
        X, y_stress, test_size=0.2, val_size=0.25, random_state=42
    )

    smote = SMOTE(random_state=42)
    os.makedirs(models_path, exist_ok=True)
    X_train_p_sm, y_train_p_sm = smote.fit_resample(X_train_p, y_train_p)
    p_model = classification_model(X_train_p_sm, X_test_p, y_train_p_sm, y_test_p)
    with open(p_model_path, 'wb') as f:
        pickle.dump(p_model, f)
    X_train_s_sm, y_train_s_sm = smote.fit_resample(X_train_s, y_train_s)
    s_model = classification_model(X_train_s_sm, X_test_s, y_train_s_sm, y_test_s)
    with open(s_model_path, 'wb') as f:
        pickle.dump(s_model, f)


if __name__ == "__main__":
    main()
