# training

Offline scripts for training, tuning, and registering the pitch-type LightGBM classifier; all experiment data is written to `mlruns/` via MLflow.

---

| File | Description |
| --- | --- |
| `config.py` | Central constants for the MLflow experiment name, LightGBM hyperparameters, Optuna search space, and temporal train/val/test split boundaries. |
| `data.py` | Loads Statcast pitch data by season, enriches it with prior-year arsenal stats, and builds the fitted `lgb.Dataset` splits and `Preprocessor` used by training and inference. |
| `predictor.py` | Defines `PitchPredictor`, an MLflow `pyfunc` model that bundles the LightGBM booster with its fitted `Preprocessor` so serving requires no refitting. |
| `evaluate.py` | Computes classification metrics (log-loss, F1, accuracy) on the test set and logs confusion matrix, feature importance, calibration, and per-class F1 plots as MLflow artifacts. |
| `promote.py` | Compares a newly registered model's log-loss against the current Production version in MLflow and transitions stages if the challenger wins. |
| `train.py` | CLI entry point for full (`--mode full`) and incremental (`--mode incremental`) training runs; trains LightGBM, logs to MLflow, and calls `promote_if_better`. |
| `tune.py` | Runs an Optuna hyperparameter search across `OPTUNA_N_TRIALS` trials (each as a nested MLflow run), then trains a final model on the best params and auto-promotes it. |
