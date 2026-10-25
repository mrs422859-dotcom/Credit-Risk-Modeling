"""Lauki Finance – Credit Risk API (FastAPI).

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .model_service import CreditRiskModel
from .schemas import (
    BatchRequest,
    BatchResponse,
    CreditApplication,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

API_VERSION = "1.0.0"
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "artifacts" / "model_data.joblib")
)
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("credit-risk-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    log.info("Loading model from %s", MODEL_PATH)
    app.state.model = CreditRiskModel(MODEL_PATH)
    log.info("Model ready (%d features)", len(app.state.model.features))
    yield


app = FastAPI(
    title="Credit Risk Model API",
    description="Predicts default probability, credit score (300–900) and rating for a loan applicant.",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model(request: Request) -> CreditRiskModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request):
    loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded, version=API_VERSION)


@app.get("/model/info", response_model=ModelInfo, tags=["meta"])
def model_info(request: Request):
    """Features, training ranges, rating bands and notebook test metrics."""
    return _model(request).info()


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(application: CreditApplication, request: Request):
    """Score a single applicant. Includes per-feature contributions and range warnings."""
    row = application.model_dump(mode="json")
    return _model(request).predict_one(row)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request):
    """Score up to 1000 applicants in one call."""
    rows = [a.model_dump(mode="json") for a in batch.applications]
    results = _model(request).predict_many(rows)
    return {
        "count": len(results),
        "results": [{"index": i, **r} for i, r in enumerate(results)],
    }
