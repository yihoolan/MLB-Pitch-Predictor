"""CLI entry point for training the pitch-type LightGBM classifier.

Usage:
    # Full training from scratch (2023+2024 train, 2024 Aug-Sep val, 2025 Apr-May test)
    python -m training.train --mode full

    # Incremental training: run once per year in October after the season ends
    python -m training.train --mode incremental --new-data-year 2026 --base-version 1
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

import pybaseball

pybaseball.cache.config.cache_directory = "data/cache"
pybaseball.cache.enable()

import argparse
from datetime import date

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from settings import settings
from training.config import (
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    N_ESTIMATORS,
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
from utils.feature_names import LABEL_COLUMN, PITCH_TYPES


def run_full() -> None:
    """Train from scratch on 2023 + partial 2024, validate on 2024 Aug–Sep, test on 2025 Apr–May."""
    print("Loading training data (2023 + 2024 Apr–Jul)...")
    train_parts = [load_season(year, end_month=end_month) for year, end_month in TRAIN_END_MONTH.items()]
    train_df = pd.concat(train_parts, ignore_index=True)

    print(f"Loading validation data ({VAL_YEAR} {VAL_MONTHS[0]:02d}–{VAL_MONTHS[1]:02d})...")
    val_df = load_season(VAL_YEAR, start_month=VAL_MONTHS[0], end_month=VAL_MONTHS[1])

    print(f"Loading test data ({TEST_YEAR} {TEST_MONTHS[0]:02d}–{TEST_MONTHS[1]:02d})...")
    test_df = load_season(TEST_YEAR, start_month=TEST_MONTHS[0], end_month=TEST_MONTHS[1])

    print("Building LightGBM datasets...")
    ds_train, ds_val, X_test, y_test, preprocessor = build_lgb_datasets(train_df, val_df, test_df)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "training_mode": "full",
                "train_years": "-".join(str(y) for y in TRAIN_END_MONTH),
                "val": f"{VAL_YEAR}-{VAL_MONTHS[0]:02d}-{VAL_MONTHS[1]:02d}",
                "test": f"{TEST_YEAR}-{TEST_MONTHS[0]:02d}-{TEST_MONTHS[1]:02d}",
            }
        )
        mlflow.log_params(
            {
                **LGBM_PARAMS,
                "n_estimators_ceiling": N_ESTIMATORS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "feature_set": "B",
                "n_features": len(preprocessor.feature_cols),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
        )

        print("Training LightGBM...")
        model = lgb.train(
            LGBM_PARAMS,
            ds_train,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[ds_val],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(50),
            ],
        )

        print("Evaluating on test set...")
        metrics = log_artifacts(model, X_test, y_test, preprocessor.feature_cols)
        print(
            f"  weighted_f1={metrics['weighted_f1']:.4f}"
            f"  log_loss={metrics['log_loss']:.4f}"
            f"  accuracy={metrics['accuracy']:.4f}"
            f"  macro_f1={metrics['macro_f1']:.4f}"
        )

        print(f"Logging model to registry as '{settings.registered_model_name}'...")
        new_version = log_predictor(model, preprocessor, registered_model_name=settings.registered_model_name)
        # Full retrains always promote — pass prod_log_loss=None to skip the challenger comparison.
        promote_if_better(new_version, metrics["log_loss"], prod_log_loss=None)
        print(f"Run complete: {run.info.run_id}")


def _score_production_on_test(test_df: pd.DataFrame, y_test: np.ndarray) -> float | None:
    """Load the current Production pyfunc and score it on test_df, returning log_loss.

    test_df must already be filtered to known pitch types (same rows that produced y_test).
    Returns None if no Production model exists — the caller will promote unconditionally.
    """
    try:
        prod_pyfunc = mlflow.pyfunc.load_model(f"models:/{settings.registered_model_name}/Production")
        prod_probs = prod_pyfunc.predict(test_df)
        return log_loss(y_test, prod_probs, labels=list(range(len(PITCH_TYPES))))
    except Exception as exc:
        print(f"  Could not load Production model for challenger eval ({exc}); promoting unconditionally.")
        return None


def run_incremental(new_data_year: int, base_version: int) -> None:
    """Append new trees on top of an existing registered model using LightGBM init_model.

    Intended to run once per year in October after the full season (Apr–Sep) is complete.
    Splits the season 70/15/15 temporally and uses challenger-based promotion: both the
    new model and the current Production model are scored on the same test slice, and
    the new model is only promoted if it wins.
    """
    today = date.today()
    if today.month < 10:
        raise ValueError(
            f"Incremental training is intended to run after the season ends. "
            f"Current month is {today.month} — re-run in October or later."
        )

    print(f"Loading base model '{settings.registered_model_name}' v{base_version} from registry...")
    model_uri = f"models:/{settings.registered_model_name}/{base_version}"
    base_model: lgb.Booster = mlflow.lightgbm.load_model(model_uri)

    print(f"Loading {new_data_year} data...")
    df_new = load_season(new_data_year)
    df_new = df_new.sort_values("game_date").reset_index(drop=True)
    n = len(df_new)
    train_df = df_new.iloc[: int(n * 0.70)].copy()
    val_df = df_new.iloc[int(n * 0.70) : int(n * 0.85)].copy()
    test_df = df_new.iloc[int(n * 0.85) :].copy()
    print(
        f"  Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows"
        f"  (val from {val_df['game_date'].iloc[0]}, test from {test_df['game_date'].iloc[0]})"
    )

    print("Building LightGBM datasets...")
    ds_train, ds_val, X_test, y_test, preprocessor = build_lgb_datasets(train_df, val_df, test_df)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "training_mode": "incremental",
                "new_data_year": str(new_data_year),
                "base_model_version": str(base_version),
                "val_start_date": str(val_df["game_date"].iloc[0]),
                "test_start_date": str(test_df["game_date"].iloc[0]),
            }
        )
        mlflow.log_params(
            {
                **LGBM_PARAMS,
                "n_estimators_ceiling": N_ESTIMATORS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "feature_set": "B",
                "n_features": len(preprocessor.feature_cols),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "base_trees": base_model.num_trees(),
            }
        )

        print(f"Incrementally training from {base_model.num_trees()} existing trees...")
        model = lgb.train(
            LGBM_PARAMS,
            ds_train,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[ds_val],
            init_model=base_model,
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(50),
            ],
        )
        mlflow.log_metric("total_trees", model.num_trees())
        mlflow.log_metric("new_trees_added", model.num_trees() - base_model.num_trees())

        print("Evaluating new model on test set...")
        metrics = log_artifacts(model, X_test, y_test, preprocessor.feature_cols)
        print(
            f"  weighted_f1={metrics['weighted_f1']:.4f}"
            f"  log_loss={metrics['log_loss']:.4f}"
            f"  accuracy={metrics['accuracy']:.4f}"
            f"  macro_f1={metrics['macro_f1']:.4f}"
        )

        print("Scoring Production model on same test slice for challenger comparison...")
        test_df_filtered = test_df[test_df[LABEL_COLUMN].isin(PITCH_TYPES)].reset_index(drop=True)
        prod_log_loss = _score_production_on_test(test_df_filtered, y_test)

        print(f"Logging updated model to registry as '{settings.registered_model_name}'...")
        new_version = log_predictor(model, preprocessor, registered_model_name=settings.registered_model_name)
        promote_if_better(new_version, metrics["log_loss"], prod_log_loss)
        print(f"Run complete: {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MLB pitch-type LightGBM classifier.")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="full: train from scratch; incremental: extend an existing model",
    )
    parser.add_argument(
        "--new-data-year", type=int, default=None, help="[incremental] Year of new season data to train on"
    )
    parser.add_argument(
        "--base-version", type=int, default=None, help="[incremental] MLflow registry version of the base model"
    )
    args = parser.parse_args()

    if args.mode == "incremental":
        if args.new_data_year is None or args.base_version is None:
            parser.error("--new-data-year and --base-version are required for incremental mode")
        run_incremental(args.new_data_year, args.base_version)
    else:
        run_full()


if __name__ == "__main__":
    main()
