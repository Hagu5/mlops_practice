"""v2: заполнить NaN в Age средним значением."""
from pathlib import Path

import pandas as pd

csv_path = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
df = pd.read_csv(csv_path)
mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)
df.to_csv(csv_path, index=False)
print(f"v2 saved: {df.shape}, Age mean={mean_age:.2f}, NaN in Age: {df['Age'].isna().sum()}")
