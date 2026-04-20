"""v1: сырой срез Titanic — Pclass, Sex, Age (с NaN в Age)."""
from pathlib import Path

from catboost.datasets import titanic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

train, _ = titanic()
subset = train[["Pclass", "Sex", "Age"]]
subset.to_csv(DATA_DIR / "titanic.csv", index=False)
print(f"v1 saved: {subset.shape}, NaN in Age: {subset['Age'].isna().sum()}")
