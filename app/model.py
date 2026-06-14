"""MLflow model registry loader with hot-reload support.

The module keeps a single ModelRegistry instance (model_registry) that the
FastAPI lifespan loads at startup and POST /reload refreshes without restarting
the server process.
"""

from __future__ import annotations

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from training.config import REGISTERED_MODEL_NAME


class ModelRegistry:
    """Holds the in-memory Production model and its registry version."""

    def __init__(self) -> None:
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.version: str | None = None

    def load_production(self) -> None:
        """Load (or reload) the Production model from the local MLflow registry.

        Raises RuntimeError if no Production version exists yet.
        """
        client = MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if not versions:
            raise RuntimeError(
                f"No Production model found in registry '{REGISTERED_MODEL_NAME}'. "
                "Run `python -m training.train --mode full` first."
            )
        self.version = versions[0].version
        uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        self.model = mlflow.pyfunc.load_model(uri)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


model_registry = ModelRegistry()
