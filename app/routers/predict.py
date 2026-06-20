from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.enrichment import enrich_row
from app.model import model_registry
from app.schemas import GameStateRequest, PitchProbabilities
from utils.feature_names import MODEL_FEATURES, PITCH_TYPES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PitchProbabilities)
def predict(request: GameStateRequest) -> PitchProbabilities:
    """Predict pitch-type probabilities for a given game state.

    Enriches the request with prior-year arsenal stats for the pitcher and
    batter (fetched once per year and cached), then runs the Production model.
    Rookie players (no prior-year stats) fall back to the training-time global
    median via the model's saved UsageImputer.
    """
    if not model_registry.is_loaded:
        logger.error("Predict called but no model is loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Call POST /reload or restart the server.")

    usage, rookie_pitcher, rookie_batter = enrich_row(
        request.pitcher_mlbam_id,
        request.batter_mlbam_id,
    )

    # Build a single-row DataFrame with all MODEL_FEATURES.
    # Base runner bools → numeric so binarize_bases sees notna() correctly.
    row: dict = {
        "balls": request.balls,
        "strikes": request.strikes,
        "outs_when_up": request.outs_when_up,
        "inning": request.inning,
        "pitch_number": request.pitch_number,
        "bat_score_diff": request.bat_score_diff,
        "on_1b": 1.0 if request.on_1b else float("nan"),
        "on_2b": 1.0 if request.on_2b else float("nan"),
        "on_3b": 1.0 if request.on_3b else float("nan"),
        "stand": request.stand,
        "p_throws": request.p_throws,
    }
    row.update(usage)

    df = pd.DataFrame([row])[MODEL_FEATURES]

    probs_array: np.ndarray = model_registry.model.predict(df)
    probs_flat: np.ndarray = np.asarray(probs_array).flatten()
    probabilities = {pt: float(p) for pt, p in zip(PITCH_TYPES, probs_flat)}
    top_pitch = max(probabilities, key=probabilities.__getitem__)

    logger.info(
        "predict pitcher=%s batter=%s top_pitch=%s",
        request.pitcher_name,
        request.batter_name,
        top_pitch,
    )

    return PitchProbabilities(
        probabilities=probabilities,
        top_pitch=top_pitch,
        pitcher_name=request.pitcher_name,
        batter_name=request.batter_name,
        rookie_pitcher=rookie_pitcher,
        rookie_batter=rookie_batter,
    )
