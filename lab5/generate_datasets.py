"""
Generate synthetic datasets for testing machine learning model quality.

This script creates:
- 3 clean datasets with clear linear relationships
- 1 noisy dataset with high variance and outliers
- Visualizations of all datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_clean_dataset(slope, intercept, n_points=150, noise_std=0.15, seed=42):
    """
    Generate a clean dataset with a linear relationship.
    
    Args:
        slope: Slope of the linear relationship
        intercept: Y-intercept of the linear relationship
        n_points: Number of data points to generate
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with 'x' and 'y' columns
    """
    np.random.seed(seed)
    x = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, noise_std, n_points)
    y = slope * x + intercept + noise
    
    return pd.DataFrame({'x': x, 'y': y})


def generate_noisy_dataset(slope, intercept, n_points=150, noise_std=3.0, 
                          outlier_fraction=0.1, seed=42):
    """
    Generate a noisy dataset with high variance and outliers.
    
    Args:
        slope: Slope of the underlying linear relationship
        intercept: Y-intercept of the underlying linear relationship
        n_points: Number of data points to generate
        noise_std: Standard deviation of Gaussian noise (high value)
        outlier_fraction: Fraction of points to make outliers
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with 'x' and 'y' columns
    """
    np.random.seed(seed)
    x = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, noise_std, n_points)
    y = slope * x + intercept + noise
    
    # Add outliers
    n_outliers = int(n_points * outlier_fraction)
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_indices] += np.random.normal(0, 10, n_outliers)
    
    return pd.DataFrame({'x': x, 'y': y})


def create_directories():
    """Create necessary directories for datasets and visualizations."""
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    print("Created directories: datasets/, visualizations/")


def save_datasets():
    """Generate and save all datasets to CSV files."""
    print("\nGenerating datasets...")
    
    # Generate clean datasets with different parameters
    clean_1 = generate_clean_dataset(slope=2.0, intercept=3.0, seed=42)
    clean_2 = generate_clean_dataset(slope=2.5, intercept=1.0, seed=43)
    clean_3 = generate_clean_dataset(slope=1.5, intercept=5.0, seed=44)
    
    # Generate noisy dataset
    noisy = generate_noisy_dataset(slope=2.0, intercept=3.0, seed=45)
    
    # Save to CSV
    clean_1.to_csv('datasets/clean_dataset_1.csv', index=False)
    clean_2.to_csv('datasets/clean_dataset_2.csv', index=False)
    clean_3.to_csv('datasets/clean_dataset_3.csv', index=False)
    noisy.to_csv('datasets/noisy_dataset.csv', index=False)
    
    print(f"[OK] Saved clean_dataset_1.csv ({len(clean_1)} points)")
    print(f"[OK] Saved clean_dataset_2.csv ({len(clean_2)} points)")
    print(f"[OK] Saved clean_dataset_3.csv ({len(clean_3)} points)")
    print(f"[OK] Saved noisy_dataset.csv ({len(noisy)} points)")
    
    return clean_1, clean_2, clean_3, noisy


def visualize_datasets(clean_1, clean_2, clean_3, noisy):
    """Create and save visualizations of all datasets."""
    print("\nCreating visualizations...")
    
    # Visualize clean datasets
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Clean Datasets', fontsize=16, fontweight='bold')
    
    datasets = [clean_1, clean_2, clean_3]
    titles = ['Clean Dataset 1\n(Training Data)', 'Clean Dataset 2', 'Clean Dataset 3']
    
    for ax, data, title in zip(axes, datasets, titles):
        ax.scatter(data['x'], data['y'], alpha=0.6, s=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/clean_datasets.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved visualizations/clean_datasets.png")
    plt.close()
    
    # Visualize noisy dataset
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(noisy['x'], noisy['y'], alpha=0.6, s=30, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Noisy Dataset with Outliers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/noisy_dataset.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved visualizations/noisy_dataset.png")
    plt.close()


def main():
    """Main function to generate all datasets and visualizations."""
    print("=" * 60)
    print("Dataset Generation Script")
    print("=" * 60)
    
    create_directories()
    clean_1, clean_2, clean_3, noisy = save_datasets()
    visualize_datasets(clean_1, clean_2, clean_3, noisy)
    
    print("\n" + "=" * 60)
    print("Dataset generation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
