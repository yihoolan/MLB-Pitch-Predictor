"""PitchPredictor: MLflow pyfunc model that bundles a LightGBM booster with its
fitted Preprocessor so inference never requires refitting from training data.

Usage (logging):
    from training.predictor import log_predictor
    version = log_predictor(model, preprocessor, registered_model_name=NAME)

Usage (loading):
    import mlflow
    predictor = mlflow.pyfunc.load_model("models:/PitchTypeClassifier/Production")
    probs = predictor.predict(game_state_df)   # shape (n_pitches, n_classes)
"""

from __future__ import annotations

import os
import pickle
import tempfile

import lightgbm as lgb
import mlflow.pyfunc
import pandas as pd

from training.data import Preprocessor
from utils.transforms import binarize_bases


class PitchPredictor(mlflow.pyfunc.PythonModel):
    """Wraps a LightGBM booster + fitted Preprocessor into a single MLflow artifact.

    load_context deserializes both from disk; predict applies the full preprocessing
    chain then returns the booster's probability matrix.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.booster = lgb.Booster(model_file=context.artifacts["booster"])
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.preprocessor: Preprocessor = pickle.load(f)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        """Transform raw game-state input and return per-class probabilities.

        model_input must contain the columns that were present at training time
        (MODEL_FEATURES). Missing usage columns are handled by the saved imputers.
        Returns a DataFrame with one column per pitch type.
        """
        prep = self.preprocessor
        work = binarize_bases(model_input)
        work = prep.pitcher_imp.transform(work)
        work = prep.batter_imp.transform(work)
        work = work.copy()
        work[prep.num_cols] = prep.num_imp.transform(work[prep.num_cols])
        for col in prep.cat_cols:
            work[col] = pd.Categorical(work[col], categories=prep.categories[col])
        X = work[prep.feature_cols].copy()
        return self.booster.predict(X)


def log_predictor(
    model: lgb.Booster,
    preprocessor: Preprocessor,
    *,
    registered_model_name: str,
) -> str:
    """Serialize booster + preprocessor and log as a pyfunc model.

    Returns the newly registered model version string.
    """
    with tempfile.TemporaryDirectory() as tmp:
        booster_path = os.path.join(tmp, "model.lgb")
        prep_path = os.path.join(tmp, "preprocessor.pkl")
        model.save_model(booster_path)
        with open(prep_path, "wb") as f:
            pickle.dump(preprocessor, f)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PitchPredictor(),
            artifacts={"booster": booster_path, "preprocessor": prep_path},
            registered_model_name=registered_model_name,
        )
    return info.registered_model_version
