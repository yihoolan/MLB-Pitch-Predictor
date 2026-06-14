"""CLI entry point for training the pitch-type LightGBM classifier.

Usage:
    # Full training from scratch (2023+2024 train, 2024 Aug-Sep val, 2025 Apr-May test)
    python -m training.train --mode full

    # Incremental training: append new trees on top of an existing registered model
    python -m training.train --mode incremental --new-data-year 2025 --base-version 1
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

import pybaseball
pybaseball.cache.enable()

import argparse

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from training.config import (
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT,
    N_ESTIMATORS,
    REGISTERED_MODEL_NAME,
    TEST_MONTHS,
    TEST_YEAR,
    TRAIN_END_MONTH,
    VAL_MONTHS,
    VAL_YEAR,
)
from training.data import build_lgb_datasets, load_season
from training.evaluate import log_artifacts


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
    ds_train, ds_val, X_test, y_test, feature_names = build_lgb_datasets(train_df, val_df, test_df)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "training_mode": "full",
            "train_years": "-".join(str(y) for y in TRAIN_END_MONTH),
            "val": f"{VAL_YEAR}-{VAL_MONTHS[0]:02d}-{VAL_MONTHS[1]:02d}",
            "test": f"{TEST_YEAR}-{TEST_MONTHS[0]:02d}-{TEST_MONTHS[1]:02d}",
        })
        mlflow.log_params({
            **LGBM_PARAMS,
            "n_estimators_ceiling": N_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "feature_set": "B",
            "n_features": len(feature_names),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        })

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
        metrics = log_artifacts(model, X_test, y_test, feature_names)
        print(
            f"  weighted_f1={metrics['weighted_f1']:.4f}"
            f"  log_loss={metrics['log_loss']:.4f}"
            f"  accuracy={metrics['accuracy']:.4f}"
            f"  macro_f1={metrics['macro_f1']:.4f}"
        )

        print(f"Logging model to registry as '{REGISTERED_MODEL_NAME}'...")
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Run complete: {run.info.run_id}")


def run_incremental(new_data_year: int, base_version: int) -> None:
    """Append new trees on top of an existing registered model using LightGBM init_model.

    Loads the base model from the MLflow registry, trains on new_data_year Apr–Jul,
    validates on new_data_year Aug–Sep, and registers the updated model as a new version.
    """
    print(f"Loading base model '{REGISTERED_MODEL_NAME}' v{base_version} from registry...")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{base_version}"
    base_model: lgb.Booster = mlflow.lightgbm.load_model(model_uri)

    print(f"Loading {new_data_year} data...")
    df_new = load_season(new_data_year)

    month = pd.to_datetime(df_new["game_date"]).dt.month
    train_df = df_new[month <= 7].copy()
    val_df = df_new[month >= 8].copy()

    print("Building LightGBM datasets...")
    # No separate test set for incremental runs — evaluate on the held-out val months
    ds_train, ds_val, X_val, y_val, feature_names = build_lgb_datasets(train_df, val_df)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "training_mode": "incremental",
            "new_data_year": str(new_data_year),
            "base_model_version": str(base_version),
        })
        mlflow.log_params({
            **LGBM_PARAMS,
            "n_estimators_ceiling": N_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "feature_set": "B",
            "n_features": len(feature_names),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "base_trees": base_model.num_trees(),
        })

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

        print("Evaluating on validation set...")
        metrics = log_artifacts(model, X_val, y_val, feature_names)
        print(
            f"  weighted_f1={metrics['weighted_f1']:.4f}"
            f"  log_loss={metrics['log_loss']:.4f}"
            f"  accuracy={metrics['accuracy']:.4f}"
            f"  macro_f1={metrics['macro_f1']:.4f}"
        )

        print(f"Logging updated model to registry as '{REGISTERED_MODEL_NAME}'...")
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Run complete: {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MLB pitch-type LightGBM classifier.")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full",
                        help="full: train from scratch; incremental: extend an existing model")
    parser.add_argument("--new-data-year", type=int, default=None,
                        help="[incremental] Year of new season data to train on")
    parser.add_argument("--base-version", type=int, default=None,
                        help="[incremental] MLflow registry version of the base model")
    args = parser.parse_args()

    if args.mode == "incremental":
        if args.new_data_year is None or args.base_version is None:
            parser.error("--new-data-year and --base-version are required for incremental mode")
        run_incremental(args.new_data_year, args.base_version)
    else:
        run_full()


if __name__ == "__main__":
    main()
