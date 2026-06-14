from __future__ import annotations

import tempfile
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from utils.feature_names import PITCH_TYPES

CALIBRATION_CLASSES = ["FF", "SI", "SL", "CH", "FC", "CU"]


def log_artifacts(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute evaluation metrics, log all artifacts to the active MLflow run.

    Logs scalar metrics directly and saves confusion matrix, classification report,
    feature importance, and per-class F1 chart as MLflow artifacts under 'eval/'.
    Returns the scalar metrics dict.
    """
    probs = model.predict(X_test)
    y_pred = probs.argmax(axis=1)

    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(PITCH_TYPES))))
    metrics: dict[str, float] = {
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_test, probs, labels=list(range(len(PITCH_TYPES)))),
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        **{f"f1_{pt}": float(score) for pt, score in zip(PITCH_TYPES, per_class_f1)},
    }
    mlflow.log_metrics(metrics)

    with tempfile.TemporaryDirectory() as tmp:
        _plot_confusion_matrix(y_test, y_pred, tmp)
        _write_classification_report(y_test, y_pred, tmp)
        _plot_feature_importance(model, feature_names, tmp)
        _plot_per_class_f1(metrics, tmp)
        _plot_calibration(y_test, probs, tmp)
        mlflow.log_artifacts(tmp, artifact_path="eval")

    return metrics


def _plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, out_dir: str) -> None:
    labels = list(range(len(PITCH_TYPES)))
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(PITCH_TYPES, rotation=45, ha="right")
    ax.set_yticklabels(PITCH_TYPES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in labels:
        for j in labels:
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > 0.6 else "black",
            )

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _write_classification_report(y_test: np.ndarray, y_pred: np.ndarray, out_dir: str) -> None:
    report = classification_report(
        y_test,
        y_pred,
        target_names=PITCH_TYPES,
        labels=list(range(len(PITCH_TYPES))),
        zero_division=0,
    )
    (Path(out_dir) / "classification_report.txt").write_text(report)


def _plot_feature_importance(model: lgb.Booster, feature_names: list[str], out_dir: str) -> None:
    importances = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.28)))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (gain)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "feature_importance.png", dpi=120)
    plt.close(fig)


def _plot_per_class_f1(metrics: dict[str, float], out_dir: str) -> None:
    f1s = [metrics[f"f1_{pt}"] for pt in PITCH_TYPES]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(PITCH_TYPES, f1s)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Pitch Type")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.axhline(
        metrics["macro_f1"], color="gray", linestyle="--", linewidth=1, label=f"macro avg = {metrics['macro_f1']:.3f}"
    )
    ax.legend(fontsize=9)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "per_class_f1.png", dpi=120)
    plt.close(fig)


def _plot_calibration(y_test: np.ndarray, probs: np.ndarray, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    for pt in CALIBRATION_CLASSES:
        idx = PITCH_TYPES.index(pt)
        binary_y = (y_test == idx).astype(int)
        frac_pos, mean_pred = calibration_curve(binary_y, probs[:, idx], n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=pt)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "calibration.png", dpi=120)
    plt.close(fig)
