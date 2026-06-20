"""FastAPI application entry point.

Run from the project root (where mlruns/ lives):
    uvicorn app.main:app --reload

Endpoints:
    GET  /health         — model load status and version
    POST /reload         — hot-swap the Production model without restarting
    GET  /players        — fuzzy player name search
    POST /predict        — pitch-type probability prediction
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from app.enrichment import _current_prior_year, _get_arsenal_tables
from app.model import model_registry
from app.routers import players, predict
from app.schemas import HealthResponse
from settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        model_registry.load_production()
    except RuntimeError:
        pass
    _get_arsenal_tables(_current_prior_year())
    yield


app = FastAPI(
    title="MLB Pitch Type Predictor",
    description="Predicts the type of the next pitch given game state and player matchup.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(players.router)
app.include_router(predict.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Return the server health status and currently loaded model version."""
    return HealthResponse(
        status="ok" if model_registry.is_loaded else "no_model",
        model_version=model_registry.version,
        model_loaded=model_registry.is_loaded,
    )


@app.post("/reload", response_model=HealthResponse, tags=["meta"])
def reload() -> HealthResponse:
    """Reload the Production model from the MLflow registry.

    Call this after running a new training run that promoted a better model.
    Takes effect immediately without restarting the server process.
    """
    model_registry.load_production()
    return HealthResponse(
        status="reloaded",
        model_version=model_registry.version,
        model_loaded=model_registry.is_loaded,
    )
