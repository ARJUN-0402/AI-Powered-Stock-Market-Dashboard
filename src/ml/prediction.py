"""Prediction result contracts for educational market-direction models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.ml.explainability import explain_prediction
from src.ml.preprocessing import TARGET_DEFINITION, FeaturePreprocessor
from src.ml.training import ModelBundle, load_model_bundle


@dataclass(frozen=True)
class PredictionResult:
    """One latest-row probabilistic direction prediction.

    ``confidence`` is the winning class probability.  It is a model output,
    not a frequentist confidence interval or a guarantee about the next bar.
    """

    symbol: str
    prediction: str
    probability: float | None
    confidence: float | None
    up_probability: float | None
    down_probability: float | None
    model_version: str | None
    model_name: str | None
    timestamp: str
    as_of: str | None
    target_definition: str = TARGET_DEFINITION
    disclaimer: str = (
        "Educational prediction, not a fact, recommendation, guarantee, or "
        "automated trading signal."
    )
    explanation: tuple[dict[str, Any], ...] = ()

    @property
    def is_available(self) -> bool:
        return self.prediction in {"up", "down"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _index_timestamp(index: pd.Index) -> str | None:
    if index is None or len(index) == 0:
        return None
    value = index[-1]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _probability_from_estimator(estimator: Any, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("estimator predict_proba must return two class columns")
        classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
        if classes.size != 2:
            raise ValueError("estimator classes_ must contain exactly two classes")
        if 1 in classes:
            up_index = int(np.flatnonzero(classes == 1)[0])
            down_index = int(np.flatnonzero(classes != 1)[0])
        elif 0 in classes:
            down_index = int(np.flatnonzero(classes == 0)[0])
            up_index = int(np.flatnonzero(classes != 0)[0])
        else:
            raise ValueError("binary estimator classes must include class 0 or 1")
        return probabilities[:, up_index], probabilities[:, down_index]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features), dtype=float)
        up = 1.0 / (1.0 + np.exp(-scores))
        return up, 1.0 - up
    raise TypeError("estimator must implement predict_proba or decision_function")


def _resolve_inputs(
    model: Any,
    features: pd.DataFrame,
    preprocessor: FeaturePreprocessor | None,
) -> tuple[Any, FeaturePreprocessor | None, pd.DataFrame, list[str]]:
    if isinstance(model, ModelBundle):
        transformed = model.preprocessor.transform(features)
        return model.estimator, model.preprocessor, transformed, list(model.feature_names)
    if isinstance(model, str):
        bundle = load_model_bundle(model)
        transformed = bundle.preprocessor.transform(features)
        return bundle.estimator, bundle.preprocessor, transformed, list(bundle.feature_names)
    if preprocessor is not None:
        transformed = preprocessor.transform(features)
        return model, preprocessor, transformed, list(preprocessor.fitted_feature_names_)
    return model, None, features, list(features.columns)


def predict(
    model: Any,
    features: pd.DataFrame,
    *,
    preprocessor: FeaturePreprocessor | None = None,
    symbol: str = "unknown",
    as_of: str | None = None,
) -> PredictionResult:
    """Predict the direction for the latest feature row without using a target."""

    timestamp = _utc_now()
    feature_timestamp = as_of or _index_timestamp(features.index if features is not None else None)
    if model is None or features is None or not isinstance(features, pd.DataFrame) or features.empty:
        return PredictionResult(
            symbol=symbol,
            prediction="unavailable",
            probability=None,
            confidence=None,
            up_probability=None,
            down_probability=None,
            model_version=getattr(getattr(model, "metadata", None), "model_version", None),
            model_name=getattr(getattr(model, "metadata", None), "model_name", None),
            timestamp=timestamp,
            as_of=feature_timestamp,
        )
    try:
        estimator, _, transformed, feature_names = _resolve_inputs(model, features, preprocessor)
        if transformed.empty:
            raise ValueError("no transformed rows are available")
        row = transformed.iloc[[-1]]
        up_probability, down_probability = _probability_from_estimator(estimator, row)
        up = float(np.clip(up_probability[0], 0.0, 1.0))
        down = float(np.clip(down_probability[0], 0.0, 1.0))
        total = up + down
        if not np.isfinite(total) or total <= 0:
            raise ValueError("estimator returned invalid class probabilities")
        up, down = up / total, down / total
        predicted = "up" if up >= down else "down"
        winning = max(up, down)
        metadata = getattr(model, "metadata", None)
        explanation: tuple[dict[str, Any], ...] = ()
        try:
            explanation = tuple(
                explain_prediction(estimator, feature_names, row, top_n=10)
            )
        except (IndexError, RuntimeError, TypeError, ValueError):
            explanation = ()
        return PredictionResult(
            symbol=symbol,
            prediction=predicted,
            probability=winning,
            confidence=winning,
            up_probability=up,
            down_probability=down,
            model_version=getattr(metadata, "model_version", None),
            model_name=getattr(metadata, "model_name", None),
            timestamp=timestamp,
            as_of=feature_timestamp,
            explanation=explanation,
        )
    except (KeyError, TypeError, ValueError):
        return PredictionResult(
            symbol=symbol,
            prediction="unavailable",
            probability=None,
            confidence=None,
            up_probability=None,
            down_probability=None,
            model_version=getattr(getattr(model, "metadata", None), "model_version", None),
            model_name=getattr(getattr(model, "metadata", None), "model_name", None),
            timestamp=timestamp,
            as_of=feature_timestamp,
        )


def predict_many(
    model: Any,
    features: pd.DataFrame,
    *,
    preprocessor: FeaturePreprocessor | None = None,
    symbol: str = "unknown",
) -> list[PredictionResult]:
    """Return one result per row; callers should normally use the latest row."""

    if features is None or features.empty:
        return []
    return [predict(model, features.iloc[[index]], preprocessor=preprocessor, symbol=symbol) for index in range(len(features))]


def predict_from_history(
    model: Any,
    data: pd.DataFrame,
    *,
    symbol: str = "unknown",
    context_frames: dict[str, pd.DataFrame] | None = None,
) -> PredictionResult:
    """Build causal latest-row features from history and predict once."""

    from src.ml.preprocessing import build_feature_frame

    features = build_feature_frame(data, context_frames=context_frames, context_lag=1)
    return predict(model, features, symbol=symbol)


def prediction_to_dict(result: PredictionResult) -> dict[str, Any]:
    return result.to_dict()
