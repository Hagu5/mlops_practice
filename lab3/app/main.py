from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import ml_model
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model.load()
    yield


app = FastAPI(title="ML Prediction Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=ml_model.is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not ml_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prediction = ml_model.predict(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(
        prediction=prediction,
        model_type=ml_model.metadata["model_type"],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if not ml_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type=ml_model.metadata["model_type"],
        target=ml_model.metadata["target"],
        features=ml_model.metadata["features"],
    )
