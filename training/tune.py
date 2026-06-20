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
import numpy as np
import optuna
import pandas as pd
from optuna_integration.lightgbm import LightGBMPruningCallback
from sklearn.metrics import accuracy_score, f1_score, log_loss

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
    trial: optuna.Trial,
    ds_train: lgb.Dataset,
    ds_val: lgb.Dataset,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    """Optuna objective: train one LightGBM candidate and return val log_loss.

    Each call runs as a nested MLflow child run so every trial's params and
    metrics are browsable under the parent study run in the MLflow UI.
    Uses LightGBMPruningCallback to abandon clearly inferior trials early.
    """
    lo, hi = OPTUNA_PARAM_SPACE["num_leaves"]
    params = {
        "objective": "multiclass",
        "num_class": len(PITCH_TYPES),
        "verbose": -1,
        "seed": RANDOM_STATE,
        "feature_pre_filter": False,
        "num_leaves": trial.suggest_int("num_leaves", lo, hi),
        "learning_rate": trial.suggest_float("learning_rate", *OPTUNA_PARAM_SPACE["learning_rate"], log=True),
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

        try:
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
        except optuna.exceptions.TrialPruned:
            mlflow.set_tag("pruned", "true")
            raise

        train_probs = model.predict(ds_train.data)
        probs = model.predict(X_val)
        y_pred = probs.argmax(axis=1)
        val_log_loss = log_loss(y_val, probs, labels=list(range(len(PITCH_TYPES))))
        mlflow.log_metrics({
            "val_log_loss": val_log_loss,
            "val_weighted_f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
            "val_accuracy": accuracy_score(y_val, y_pred),
            "train_log_loss": log_loss(ds_train.label, train_probs, labels=list(range(len(PITCH_TYPES)))),
            "num_trees": model.num_trees(),
            "best_iteration": model.best_iteration,
        })

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


def run_tuning() -> None:
    """Run the full Optuna search, train the final model, register, and auto-promote."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, val_df, test_df = _load_splits()

    print("Building LightGBM datasets for Optuna search...")
    ds_train, ds_val, X_val, y_val, _ = build_lgb_datasets(train_df, val_df)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=OPTUNA_STUDY_NAME) as parent_run:
        mlflow.log_param("n_trials", OPTUNA_N_TRIALS)

        print(f"Starting Optuna study ({OPTUNA_N_TRIALS} trials)...")
        study = optuna.create_study(direction="minimize", study_name=OPTUNA_STUDY_NAME)
        study.optimize(
            lambda trial: _objective(trial, ds_train, ds_val, X_val, y_val),
            n_trials=OPTUNA_N_TRIALS,
            show_progress_bar=True,
        )

        best = study.best_trial
        print(f"Best trial #{best.number}: val_log_loss={best.value:.4f}")
        print(f"  params: {best.params}")
        mlflow.log_metric("best_val_log_loss", best.value)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

        # Train final model on train+val combined so every labelled example is used
        print("Training final model on train+val combined...")
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        ds_final, _, X_test, y_test, preprocessor = build_lgb_datasets(combined_df, test_df)

        best_params = {
            **LGBM_PARAMS,
            **best.params,
        }
        best_n_iters = best.user_attrs.get("best_iteration", N_ESTIMATORS)

        final_model = lgb.train(
            best_params,
            ds_final,
            num_boost_round=best_n_iters,
            callbacks=[lgb.log_evaluation(50)],
        )

        print("Evaluating final model on test set...")
        metrics = log_artifacts(final_model, X_test, y_test, preprocessor.feature_cols)
        print(
            f"  weighted_f1={metrics['weighted_f1']:.4f}"
            f"  log_loss={metrics['log_loss']:.4f}"
            f"  accuracy={metrics['accuracy']:.4f}"
            f"  macro_f1={metrics['macro_f1']:.4f}"
        )
        mlflow.log_params(
            {
                **best_params,
                "n_features": len(preprocessor.feature_cols),
                "train_rows": len(combined_df),
                "test_rows": len(test_df),
                "final_n_estimators": best_n_iters,
            }
        )

        print(f"Registering final model as '{REGISTERED_MODEL_NAME}'...")
        new_version = log_predictor(final_model, preprocessor, registered_model_name=REGISTERED_MODEL_NAME)
        # Tuning runs always promote — pass prod_log_loss=None to skip the challenger comparison.
        promote_if_better(new_version, metrics["log_loss"], prod_log_loss=None)
        print(f"Tuning run complete: {parent_run.info.run_id}")


if __name__ == "__main__":
    run_tuning()
