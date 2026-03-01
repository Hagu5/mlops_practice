#!/bin/bash

# Скрипт установки зависимостей для data_creation.py в виртуальное окружение

set -e  # Остановка при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"



# Создание виртуального окружения
python3 -m venv "$VENV_DIR"

# Активация виртуального окружения
source "$VENV_DIR/bin/activate"



# Обновление pip
pip install --upgrade pip

# Установка kagglehub
pip install kagglehub

# Установка pandas и scikit-learn
pip install pandas scikit-learn imbalanced-learn xgboost lightgbm



# Запуск пайплайна в виртуальном окружении
python3 data_creation.py

python3 data_preprocessing.py

python3 model_preparation.py

python3 model_testing.py

