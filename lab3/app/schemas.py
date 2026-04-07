from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: float
    model_type: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    target: str
    features: list[str]
