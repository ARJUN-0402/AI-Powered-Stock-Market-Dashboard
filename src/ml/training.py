"""Reproducible, time-aware training and model persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.ml.evaluation import classification_metrics, compare_model_metrics
from src.ml.preprocessing import (
    FEATURE_GROUPS,
    TARGET_DEFINITION,
    FeaturePreprocessor,
    align_features_target,
    time_aware_split,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - dependency is declared by the project
    XGBClassifier = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata required to interpret and reproduce a model artifact."""

    schema_version: str = "1.0"
    model_name: str = ""
    model_version: str = ""
    target_definition: str = TARGET_DEFINITION
    feature_names: tuple[str, ...] = ()
    feature_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    trained_at: str = ""
    training_start: str | None = None
    training_end: str | None = None
    train_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0
    random_state: int = 42
    selection_metric: str = "balanced_accuracy"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    libraries: dict[str, str] = field(default_factory=dict)
    disclaimer: str = (
        "Educational probabilistic prediction only; not a fact, recommendation, "
        "guarantee, or automated trading signal."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBundle:
    """Fitted estimator, fitted preprocessing state, and immutable metadata."""

    estimator: Any
    preprocessor: FeaturePreprocessor
    metadata: ModelMetadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata.feature_names)


@dataclass(frozen=True)
class ModelComparison:
    model_name: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    selected: bool = False


@dataclass
class TrainingResult:
    bundles: dict[str, ModelBundle]
    comparisons: dict[str, ModelComparison]
    selected_model_name: str
    selection: dict[str, Any]
    split_info: dict[str, Any]

    @property
    def selected_bundle(self) -> ModelBundle:
        return self.bundles[self.selected_model_name]


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    n_estimators: int = 250
    max_depth: int = 4
    learning_rate: float = 0.05
    primary_metric: str = "balanced_accuracy"
    calibrate: bool = False
    calibration_method: str = "sigmoid"
    calibration_splits: int = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except ImportError:
        versions["scikit-learn"] = "unavailable"
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        versions["xgboost"] = "unavailable"
    return versions
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost_version,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _model_factories(config: TrainingConfig) -> dict[str, Any]:
    common = {"random_state": config.random_state}
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            **common,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            class_weight="balanced_subsample",
            n_jobs=-1,
            **common,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            **common,
        )
    return models


def _balanced_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def _version_for(
    model_name: str,
    feature_names: list[str],
    config: TrainingConfig,
    train_index: pd.Index,
) -> str:
    payload = {
        "model_name": model_name,
        "feature_names": feature_names,
        "config": asdict(config),
        "training_start": str(train_index[0]),
        "training_end": str(train_index[-1]),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{model_name}-v1-{digest}"


def _fit_with_optional_calibration(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig,
) -> Any:
    if not config.calibrate:
        estimator.fit(features, target)
        return estimator
    if target.nunique() < 2:
        estimator.fit(features, target)
        return estimator
    splits = min(config.calibration_splits, max(2, len(features) // 20))
    if splits < 2:
        estimator.fit(features, target)
        return estimator
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=config.calibration_method,
        cv=TimeSeriesSplit(n_splits=splits),
    )
    calibrated.fit(features, target)
    return calibrated


def _metrics_with_balanced_accuracy(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    predictions = np.asarray(estimator.predict(features))
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(features))[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features))
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        probabilities = None
    metrics = classification_metrics(target, predictions, probabilities)
    metrics["balanced_accuracy"] = _balanced_accuracy(target, predictions)
    return metrics


def train_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    split: tuple[Any, ...] | None = None,
    preprocessor: FeaturePreprocessor | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train and compare all supported classifiers on one temporal split."""

    config = config or TrainingConfig()
    aligned_features, aligned_target = align_features_target(features, target)
    if aligned_features.empty:
        raise ValueError("no complete target rows are available for training")
    if split is None:
        split = time_aware_split(
            aligned_features,
            aligned_target,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            gap_rows=1,
            min_train_rows=20,
        )
    if len(split) != 6:
        raise ValueError("split must contain train/validation/test features and targets")
    train_features, validation_features, test_features, train_target, validation_target, test_target = split
    if train_features.empty or validation_features.empty or test_features.empty:
        raise ValueError("train, validation, and test splits must be non-empty")
    if train_target.nunique() < 2:
        raise ValueError("training target must contain both direction classes")

    preprocessor = preprocessor or FeaturePreprocessor()
    train_transformed = preprocessor.fit_transform(train_features)
    validation_transformed = preprocessor.transform(validation_features)
    test_transformed = preprocessor.transform(test_features)
    factories = _model_factories(config)
    bundles: dict[str, ModelBundle] = {}
    comparisons: dict[str, ModelComparison] = {}
    selection_results: dict[str, dict[str, Any]] = {}

    for model_name, factory in factories.items():
        estimator = _fit_with_optional_calibration(
            clone(factory), train_transformed, train_target, config
        )
        validation_metrics = _metrics_with_balanced_accuracy(
            estimator, validation_transformed, validation_target
        )
        test_metrics = _metrics_with_balanced_accuracy(
            estimator, test_transformed, test_target
        )
        baseline = validation_metrics["baseline"]
        metadata = ModelMetadata(
            model_name=model_name,
            model_version=_version_for(
                model_name, preprocessor.fitted_feature_names_, config, train_features.index
            ),
            feature_names=tuple(preprocessor.fitted_feature_names_),
            feature_groups=dict(FEATURE_GROUPS),
            trained_at=_utc_now(),
            training_start=str(train_features.index.min()),
            training_end=str(train_features.index.max()),
            train_rows=len(train_features),
            validation_rows=len(validation_features),
            test_rows=len(test_features),
            random_state=config.random_state,
            selection_metric=config.primary_metric,
            hyperparameters=_jsonable(factory.get_params(deep=False)),
            validation_metrics=_jsonable(validation_metrics),
            test_metrics=_jsonable(test_metrics),
            baseline_metrics=_jsonable(baseline),
            libraries=_library_versions(),
        )
        bundle = ModelBundle(estimator, preprocessor, metadata)
        bundles[model_name] = bundle
        comparisons[model_name] = ModelComparison(
            model_name=model_name,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            baseline_metrics=baseline,
        )
        selection_results[model_name] = {
            config.primary_metric: validation_metrics.get(
                config.primary_metric,
                validation_metrics.get("balanced_accuracy", validation_metrics.get("f1")),
            ),
            "balanced_accuracy": validation_metrics["balanced_accuracy"],
            "f1": validation_metrics["f1"],
            "roc_auc": validation_metrics.get("roc_auc"),
        }

    selection = compare_model_metrics(selection_results, primary_metric=config.primary_metric)
    selected_name = selection["selected_model"]
    for name, comparison in comparisons.items():
        comparison.selected = name == selected_name
    split_info = {
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "train_start": str(train_features.index.min()),
        "train_end": str(train_features.index.max()),
        "validation_start": str(validation_features.index.min()),
        "validation_end": str(validation_features.index.max()),
        "test_start": str(test_features.index.min()),
        "test_end": str(test_features.index.max()),
        "gap_rows": 1,
    }
    return TrainingResult(bundles, comparisons, selected_name, selection, split_info)


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model: Any | None = None,
    *,
    preprocessor: FeaturePreprocessor | None = None,
    random_state: int = 42,
) -> Any:
    """Fit one estimator, or return the default logistic regression fit.

    An arbitrary object without a ``fit`` method is returned unchanged for
    backwards compatibility with the original thin interface.
    """

    if features is None or target is None or features.empty or target.empty:
        return None if model is None else model
    aligned_features, aligned_target = align_features_target(features, target)
    if aligned_features.empty:
        return None if model is None else model
    estimator = model
    if estimator is None:
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )
    if not hasattr(estimator, "fit"):
        return estimator
    transformed = aligned_features
    if preprocessor is not None:
        transformed = preprocessor.fit_transform(aligned_features)
    estimator.fit(transformed, aligned_target)
    return estimator


def train_pipeline(
    data: pd.DataFrame,
    *,
    context_frames: Mapping[str, pd.DataFrame] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Run feature engineering, target construction, splitting, and training."""

    from src.ml.preprocessing import build_feature_frame, build_target

    features = build_feature_frame(
        data,
        windows=(5, 20, 50),
        context_frames=context_frames,
        context_lag=1,
    )
    target = build_target(data, horizon=1)
    result = train_models(features, target, config=config)
    if output_path is not None:
        save_model_bundle(result.selected_bundle, output_path)
    return result


def save_model_bundle(bundle: ModelBundle, path: str | os.PathLike[str]) -> Path:
    """Atomically persist a model bundle and a human-readable metadata sidecar."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, destination)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(
        json.dumps(bundle.metadata.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return destination


def load_model_bundle(path: str | os.PathLike[str]) -> ModelBundle:
    """Load a trusted joblib artifact and validate its metadata schema."""

    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("artifact is not a ModelBundle")
    if not isinstance(bundle.metadata, ModelMetadata):
        raise TypeError("artifact metadata is missing or invalid")
    if bundle.metadata.schema_version != "1.0":
        raise ValueError(f"unsupported model schema: {bundle.metadata.schema_version}")
    return bundle
