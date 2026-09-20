"""Evaluation metrics for leakage-safe market-direction classifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_array(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(list(values))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _finite_probabilities(probabilities: Iterable[float], size: int) -> np.ndarray:
    values = np.asarray(list(probabilities), dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        values = values[:, 1]
    if values.shape != (size,):
        raise ValueError("probabilities must contain one value per observation")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    return np.clip(values, 1e-6, 1 - 1e-6)


def classification_metrics(
    y_true: Iterable[int | bool],
    y_pred: Iterable[int | bool],
    y_proba: Iterable[float] | None = None,
    *,
    positive_label: int = 1,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Return classification, ranking, probability, and baseline metrics.

    ROC-AUC is returned as ``None`` when a split contains only one class.  This
    is an explicit unavailable result rather than a fabricated score.
    """

    true = _as_array(y_true, "y_true").astype(int)
    pred = _as_array(y_pred, "y_pred").astype(int)
    if true.size == 0:
        raise ValueError("y_true must contain at least one observation")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")

    labels = np.array([0, positive_label], dtype=int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(true, pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(true, pred, pos_label=positive_label, zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).astype(int).tolist(),
        "class_distribution": {
            str(int(label)): int(np.sum(true == label)) for label in np.unique(true)
        },
        "roc_auc": None,
        "log_loss": None,
        "brier_score": None,
        "calibration": None,
    }

    if y_proba is not None:
        probabilities = _finite_probabilities(y_proba, true.size)
        metrics["roc_auc"] = (
            float(roc_auc_score(true, probabilities)) if np.unique(true).size >= 2 else None
        )
        metrics["log_loss"] = float(log_loss(true, probabilities, labels=labels))
        metrics["brier_score"] = float(brier_score_loss((true == positive_label).astype(int), probabilities))
        metrics["calibration"] = calibration_metrics(true, probabilities, calibration_bins)

    majority = int(np.bincount(true, minlength=2).argmax())
    baseline_pred = np.full_like(true, majority)
    baseline = {
        "majority_class": majority,
        "accuracy": float(accuracy_score(true, baseline_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, baseline_pred)),
        "precision": float(
            precision_score(true, baseline_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(true, baseline_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(true, baseline_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": None,
    }
    metrics["baseline"] = baseline
    return metrics


def calibration_metrics(
    y_true: Iterable[int | bool],
    probabilities: Iterable[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calculate expected calibration error and a reliability table."""

    true = _as_array(y_true, "y_true").astype(int)
    probability = _finite_probabilities(probabilities, true.size)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(probability, edges[1:-1], right=True)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        count = int(mask.sum())
        if count:
            mean_prediction = float(probability[mask].mean())
            observed = float((true[mask] == 1).mean())
            ece += count / true.size * abs(observed - mean_prediction)
        else:
            mean_prediction = None
            observed = None
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_predicted_probability": mean_prediction,
                "observed_frequency": observed,
            }
        )
    return {
        "expected_calibration_error": float(ece),
        "n_bins": int(n_bins),
        "reliability": rows,
    }


def evaluate_classifier(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Evaluate an already-fitted classifier and return JSON-safe metrics."""

    if features.empty or target.empty:
        raise ValueError("features and target must be non-empty")
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        probabilities = None
    predictions = estimator.predict(features)
    return classification_metrics(target, predictions, probabilities)


def compare_model_metrics(
    results: Mapping[str, Mapping[str, Any]],
    *,
    primary_metric: str = "balanced_accuracy",
) -> dict[str, Any]:
    """Select a model by validation evidence, never by test performance."""

    if not results:
        raise ValueError("at least one model result is required")
    available = [name for name, metrics in results.items() if primary_metric in metrics]
    if not available:
        primary_metric = "f1"
        available = [name for name, metrics in results.items() if primary_metric in metrics]
    if not available:
        raise ValueError("none of the model results contain a selection metric")
    selected = max(available, key=lambda name: float(results[name][primary_metric]))
    return {
        "selected_model": selected,
        "primary_metric": primary_metric,
        "validation_scores": {
            name: float(results[name][primary_metric]) for name in available
        },
    }


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Mapping[str, float]:
    """Return MSE, MAE, and R2 for backwards-compatible regression callers."""

    true = list(y_true)
    pred = list(y_pred)
    if not true or not pred or len(true) != len(pred):
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    n = len(true)
    errors = [left - right for left, right in zip(true, pred, strict=False)]
    mse = sum(error * error for error in errors) / n
    mae = sum(abs(error) for error in errors) / n
    mean_true = sum(true) / n
    total = sum((value - mean_true) ** 2 for value in true)
    residual = sum(error * error for error in errors)
    r2 = 1.0 - residual / total if total else 0.0
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
