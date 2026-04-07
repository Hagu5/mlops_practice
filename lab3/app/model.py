import json
import os
import pickle

import numpy as np
import pandas as pd

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))


class MLModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None

    def load(self):
        with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
            self.metadata = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: dict[str, float]) -> float:
        expected = self.metadata["features"]
        missing = set(expected) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df = pd.DataFrame([{col: features[col] for col in expected}])
        scaled = self.scaler.transform(df)
        prediction = self.model.predict(scaled)
        return float(prediction[0])


ml_model = MLModel()
