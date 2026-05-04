"""
Train a linear regression model on clean dataset.

This script:
- Loads the first clean dataset
- Trains a scikit-learn LinearRegression model
- Saves the trained model using pickle
- Prints training metrics
"""

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os


def load_training_data():
    """
    Load the training dataset.
    
    Returns:
        X, y: Feature matrix and target vector
    """
    if not os.path.exists('datasets/clean_dataset_1.csv'):
        raise FileNotFoundError(
            "Training dataset not found. Please run generate_datasets.py first."
        )
    
    data = pd.read_csv('datasets/clean_dataset_1.csv')
    X = data[['x']].values
    y = data['y'].values
    
    print(f"Loaded training data: {len(data)} samples")
    return X, y


def train_model(X, y):
    """
    Train a linear regression model.
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        Trained model
    """
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"\nModel trained successfully!")
    print(f"Coefficients: slope = {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    return model


def evaluate_model(model, X, y):
    """
    Evaluate the trained model on training data.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
    """
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print("\n" + "=" * 60)
    print("Training Metrics:")
    print("=" * 60)
    print(f"R2 Score:              {r2:.6f}")
    print(f"Mean Squared Error:    {mse:.6f}")
    print(f"Mean Absolute Error:   {mae:.6f}")
    print("=" * 60)


def save_model(model, filename='model.pkl'):
    """
    Save the trained model to a pickle file.
    
    Args:
        model: Trained model
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n[OK] Model saved to {filename}")


def main():
    """Main function to train and save the model."""
    print("=" * 60)
    print("Model Training Script")
    print("=" * 60)
    print()
    
    # Load data
    X, y = load_training_data()
    
    # Train model
    model = train_model(X, y)
    
    # Evaluate on training data
    evaluate_model(model, X, y)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
