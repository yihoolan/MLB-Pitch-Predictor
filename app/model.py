"""Model loader for the FastAPI service.

Loads the LightGBM booster and fitted Preprocessor directly from the committed
model/ directory. No MLflow registry access is needed at serve time.

The module keeps a single ModelRegistry instance (model_registry) that the
FastAPI lifespan loads at startup and POST /reload refreshes without restarting
the server process.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb

from training.predictor import PitchPredictor

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model"


class _LoadedModel:
    """Thin wrapper that presents a .predict(df) interface over PitchPredictor."""

    def __init__(self, predictor: PitchPredictor) -> None:
        self._predictor = predictor

    def predict(self, df):
        # PitchPredictor.predict(context, model_input) — context is unused at inference
        return self._predictor.predict(None, df)


class ModelRegistry:
    """Holds the in-memory model and its version string."""

    def __init__(self) -> None:
        self.model: _LoadedModel | None = None
        self.version: str | None = None

    def load_production(self) -> None:
        """Load (or reload) the model from the model/ directory.

        Raises RuntimeError if the model files are missing.
        """
        booster_path = MODEL_DIR / "model.lgb"
        preprocessor_path = MODEL_DIR / "preprocessor.pkl"
        version_path = MODEL_DIR / "version.txt"

        if not booster_path.exists() or not preprocessor_path.exists():
            raise RuntimeError(
                "Model files not found in model/. "
                "Run `python scripts/export_model.py` to export the Production model first."
            )

        predictor = PitchPredictor()
        predictor.booster = lgb.Booster(model_file=str(booster_path))
        with open(preprocessor_path, "rb") as f:
            predictor.preprocessor = pickle.load(f)

        self.model = _LoadedModel(predictor)
        self.version = version_path.read_text().strip() if version_path.exists() else "unknown"
        logger.info("Model loaded successfully (version=%s)", self.version)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


model_registry = ModelRegistry()
