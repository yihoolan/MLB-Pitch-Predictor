"""One-time Optuna hyperparameter search for the pitch-type LightGBM classifier.

Runs OPTUNA_N_TRIALS trials, each as a nested MLflow child run under a single
parent run.  The best trial's parameters are used to train a final model on the
combined train+val split, which is then registered and auto-promoted if it
beats the current Production model on log_loss.

Usage:
    python -m training.tune
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

import pybaseball

pybaseball.cache.enable()

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import log_loss

from training.config import (
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT,
    N_ESTIMATORS,
    OPTUNA_N_TRIALS,
    OPTUNA_PARAM_SPACE,
    OPTUNA_STUDY_NAME,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    TEST_MONTHS,
    TEST_YEAR,
    TRAIN_END_MONTH,
    VAL_MONTHS,
    VAL_YEAR,
)
from training.data import build_lgb_datasets, load_season
from training.evaluate import log_artifacts
from training.predictor import log_predictor
from training.promote import promote_if_better
from utils.feature_names import PITCH_TYPES


def _objective(
    trial: "optuna.Trial",
    ds_train: lgb.Dataset,
    ds_val: lgb.Dataset,
    X_val: "pd.DataFrame",
    y_val: "np.ndarray",
) -> float:
    """Optuna objective: train one LightGBM candidate and return val log_loss.

    Each call runs as a nested MLflow child run so every trial's params and
    metrics are browsable under the parent study run in the MLflow UI.
    Uses LightGBMPruningCallback to abandon clearly inferior trials early.
    """
    import numpy as np
    import optuna
    from optuna_integration.lightgbm import LightGBMPruningCallback

    lo, hi = OPTUNA_PARAM_SPACE["num_leaves"]
    params = {
        "objective": "multiclass",
        "num_class": len(PITCH_TYPES),
        "verbose": -1,
        "seed": RANDOM_STATE,
        "num_leaves": trial.suggest_int("num_leaves", lo, hi),
        "learning_rate": trial.suggest_float(
            "learning_rate", *OPTUNA_PARAM_SPACE["learning_rate"], log=True
        ),
        "min_child_samples": trial.suggest_int("min_child_samples", *OPTUNA_PARAM_SPACE["min_child_samples"]),
        "feature_fraction": trial.suggest_float("feature_fraction", *OPTUNA_PARAM_SPACE["feature_fraction"]),
        "bagging_fraction": trial.suggest_float("bagging_fraction", *OPTUNA_PARAM_SPACE["bagging_fraction"]),
        "bagging_freq": trial.suggest_int("bagging_freq", *OPTUNA_PARAM_SPACE["bagging_freq"]),
        "lambda_l1": trial.suggest_float("lambda_l1", *OPTUNA_PARAM_SPACE["lambda_l1"]),
        "lambda_l2": trial.suggest_float("lambda_l2", *OPTUNA_PARAM_SPACE["lambda_l2"]),
    }

    tunable_keys = set(OPTUNA_PARAM_SPACE)
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({k: v for k, v in params.items() if k in tunable_keys})

        model = lgb.train(
            params,
            ds_train,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[ds_val],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
                LightGBMPruningCallback(trial, "multi_logloss"),
            ],
        )

        probs = model.predict(X_val)
        val_log_loss = log_loss(y_val, probs, labels=list(range(len(PITCH_TYPES))))
        mlflow.log_metric("val_log_loss", val_log_loss)
        mlflow.log_metric("best_iteration", model.best_iteration)

    trial.set_user_attr("best_iteration", model.best_iteration)
    return val_log_loss


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return (train_df, val_df, test_df) using the same temporal split as run_full."""
    print("Loading training data (2023 + 2024 Apr–Jul)...")
    train_parts = [load_season(year, end_month=end_month) for year, end_month in TRAIN_END_MONTH.items()]
    train_df = pd.concat(train_parts, ignore_index=True)

    print(f"Loading validation data ({VAL_YEAR} {VAL_MONTHS[0]:02d}–{VAL_MONTHS[1]:02d})...")
    val_df = load_season(VAL_YEAR, start_month=VAL_MONTHS[0], end_month=VAL_MONTHS[1])

    print(f"Loading test data ({TEST_YEAR} {TEST_MONTHS[0]:02d}–{TEST_MONTHS[1]:02d})...")
    test_df = load_season(TEST_YEAR, start_month=TEST_MONTHS[0], end_month=TEST_MONTHS[1])

    return train_df, val_df, test_df
