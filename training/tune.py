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
