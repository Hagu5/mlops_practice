"""
Pytest tests for machine learning model quality.

This module contains tests to verify model performance on:
- Clean datasets (should pass)
- Noisy dataset (should fail/show poor metrics)
"""

import pytest
import pandas as pd
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


@pytest.fixture(scope="module")
def trained_model():
    """
    Load the trained model for all tests.
    
    Returns:
        Trained LinearRegression model
    """
    model_path = 'model.pkl'
    if not os.path.exists(model_path):
        pytest.fail(f"Model file not found: {model_path}. Run train_model.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def load_dataset(filename):
    """
    Load a dataset from CSV file.
    
    Args:
        filename: Path to CSV file
    
    Returns:
        X, y: Feature matrix and target vector
    """
    filepath = f'datasets/{filename}'
    if not os.path.exists(filepath):
        pytest.fail(f"Dataset not found: {filepath}. Run generate_datasets.py first.")
    
    data = pd.read_csv(filepath)
    X = data[['x']].values
    y = data['y'].values
    
    return X, y


def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Dictionary with R², MSE, and MAE
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def test_model_on_clean_dataset_1(trained_model):
    """
    Test model on the first clean dataset (training data).
    
    Expected: High R² (>0.85), Low MSE (<1.0), Low MAE (<0.5)
    """
    X, y = load_dataset('clean_dataset_1.csv')
    y_pred = trained_model.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    
    print(f"\nClean Dataset 1 Metrics:")
    print(f"  R² Score: {metrics['r2']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Assertions for clean data
    assert metrics['r2'] > 0.85, f"R² score {metrics['r2']:.4f} is below threshold 0.85"
    assert metrics['mse'] < 1.0, f"MSE {metrics['mse']:.4f} is above threshold 1.0"
    assert metrics['mae'] < 0.5, f"MAE {metrics['mae']:.4f} is above threshold 0.5"


def test_model_on_clean_dataset_2(trained_model):
    """
    Test model on the second clean dataset.
    
    Expected: Good R2 (>0.80), Reasonable MSE (<5.0), Reasonable MAE (<2.0)
    Note: Different slope/intercept than training data, so metrics will be lower.
    """
    X, y = load_dataset('clean_dataset_2.csv')
    y_pred = trained_model.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    
    print(f"\nClean Dataset 2 Metrics:")
    print(f"  R2 Score: {metrics['r2']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Assertions for clean data with different parameters
    assert metrics['r2'] > 0.80, f"R2 score {metrics['r2']:.4f} is below threshold 0.80"
    assert metrics['mse'] < 5.0, f"MSE {metrics['mse']:.4f} is above threshold 5.0"
    assert metrics['mae'] < 2.0, f"MAE {metrics['mae']:.4f} is above threshold 2.0"


def test_model_on_clean_dataset_3(trained_model):
    """
    Test model on the third clean dataset.
    
    Expected: Good R2 (>0.80), Reasonable MSE (<5.0), Reasonable MAE (<2.0)
    Note: Different slope/intercept than training data, so metrics will be lower.
    """
    X, y = load_dataset('clean_dataset_3.csv')
    y_pred = trained_model.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    
    print(f"\nClean Dataset 3 Metrics:")
    print(f"  R2 Score: {metrics['r2']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Assertions for clean data with different parameters
    assert metrics['r2'] > 0.80, f"R2 score {metrics['r2']:.4f} is below threshold 0.80"
    assert metrics['mse'] < 5.0, f"MSE {metrics['mse']:.4f} is above threshold 5.0"
    assert metrics['mae'] < 2.0, f"MAE {metrics['mae']:.4f} is above threshold 2.0"


def test_model_on_noisy_dataset(trained_model):
    """
    Test model on the noisy dataset with outliers.
    
    Expected: This test should FAIL due to poor metrics on noisy data.
    The model will have low R2, high MSE, and high MAE.
    """
    X, y = load_dataset('noisy_dataset.csv')
    y_pred = trained_model.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    
    print(f"\nNoisy Dataset Metrics:")
    print(f"  R2 Score: {metrics['r2']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Assertions for clean data (will fail on noisy data)
    assert metrics['r2'] > 0.80, f"R2 score {metrics['r2']:.4f} is below threshold 0.80 - NOISY DATA DETECTED"
    assert metrics['mse'] < 5.0, f"MSE {metrics['mse']:.4f} is above threshold 5.0 - NOISY DATA DETECTED"
    assert metrics['mae'] < 2.0, f"MAE {metrics['mae']:.4f} is above threshold 2.0 - NOISY DATA DETECTED"
