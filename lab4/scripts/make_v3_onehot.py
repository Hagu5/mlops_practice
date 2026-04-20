"""v3: one-hot encoding для признака Sex."""
from pathlib import Path

import pandas as pd

csv_path = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
df = pd.read_csv(csv_path)
onehot = pd.get_dummies(df["Sex"], prefix="Sex", dtype=int)
df = pd.concat([df.drop(columns=["Sex"]), onehot], axis=1)
df.to_csv(csv_path, index=False)
print(f"v3 saved: {df.shape}, columns: {df.columns.tolist()}")
