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
            f"  accuracy={metrics['accuracy']:.4f}"
            f"  macro_f1={metrics['macro_f1']:.4f}"
            f"  weighted_f1={metrics['weighted_f1']:.4f}"
        )

        print(f"Logging model to registry as '{REGISTERED_MODEL_NAME}'...")
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Run complete: {run.info.run_id}")
