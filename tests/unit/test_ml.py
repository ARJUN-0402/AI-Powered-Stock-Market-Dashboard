"""Tests for the leakage-safe market-direction ML pipeline."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.ml.evaluation import (
    calibration_metrics,
    classification_metrics,
    compare_model_metrics,
    evaluate_classifier,
    regression_metrics,
)
from src.ml.explainability import (
    explain_model,
    explain_prediction,
    feature_importance,
)
from src.ml.prediction import (
    PredictionResult,
    predict,
    predict_from_history,
    predict_many,
    prediction_to_dict,
)
from src.ml.preprocessing import (
    FEATURE_GROUPS,
    TARGET_DEFINITION,
    FeaturePreprocessor,
    align_features_target,
    build_classification_target,
    build_feature_frame,
    build_target,
    create_target,
    prepare_dataset,
    time_aware_split,
    train_test_split,
)
from src.ml.training import (
    ModelBundle,
    ModelMetadata,
    TrainingConfig,
    TrainingResult,
    _version_for,
    load_model_bundle,
    save_model_bundle,
    train_model,
    train_models,
    train_pipeline,
)
from tests.fixtures import make_price_frame


def _price_frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": closes},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def _ohlcv_frame(closes: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    opens = closes.to_numpy() + rng.normal(0.0, 0.2, len(closes))
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes.to_numpy()) + 0.5,
            "Low": np.minimum(opens, closes.to_numpy()) - 0.5,
            "Close": closes.to_numpy(),
            "Volume": rng.integers(100_000, 500_000, len(closes)),
        },
        index=closes.index,
    )


def _volatile_frame(rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    log_returns = rng.normal(0.0002, 0.015, rows)
    closes = pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=index, dtype=float)
    return _ohlcv_frame(closes)


@pytest.fixture(scope="module")
def aligned_dataset() -> tuple[pd.DataFrame, pd.Series]:
    data = _volatile_frame()
    features = build_feature_frame(data, windows=(5, 20, 50))
    target = build_target(data, horizon=1, threshold=0.0)
    return align_features_target(features, target)


@pytest.fixture(scope="module")
def trained_result() -> TrainingResult:
    config = TrainingConfig(n_estimators=15, max_depth=3)
    return train_pipeline(_volatile_frame(), config=config)


def test_target_generation_binary_direction_and_definition() -> None:
    data = _price_frame([100.0, 101.0, 99.0, 102.0])
    target = build_target(data)

    assert target.tolist()[:3] == [1.0, 0.0, 1.0]
    assert pd.isna(target.iloc[-1])
    assert target.name == "target_1d"
    assert target.attrs["horizon"] == 1
    assert target.attrs["definition"] == TARGET_DEFINITION
    assert create_target is build_target
    assert build_classification_target is build_target


def test_target_generation_uses_requested_horizon() -> None:
    data = _price_frame([100.0, 101.0, 99.0, 102.0, 103.0])
    target = build_target(data, horizon=2)

    assert target.iloc[:3].tolist() == [0.0, 1.0, 1.0]
    assert target.iloc[3:].isna().all()
    assert target.name == "target_2d"


def test_target_generation_threshold_creates_neutral_band() -> None:
    data = _price_frame([100.0, 102.0, 99.96, 100.05, 99.95])
    target = build_target(data, threshold=0.01)

    assert target.iloc[0] == 1.0
    assert target.iloc[1] == 0.0
    assert target.iloc[2:4].isna().all()
    assert pd.isna(target.iloc[-1])


def test_target_generation_handles_zero_price_without_infinity() -> None:
    data = _price_frame([100.0, 0.0, 110.0, 99.0])
    target = build_target(data)

    assert target.iloc[0] == 0.0
    assert pd.isna(target.iloc[1])
    assert target.iloc[2] == 0.0
    assert pd.isna(target.iloc[-1])
    assert np.isfinite(target.dropna()).all()


@pytest.mark.parametrize("horizon", [0, -1])
def test_target_generation_rejects_invalid_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        build_target(_price_frame([100.0, 101.0]), horizon=horizon)


@pytest.mark.parametrize("threshold", [-0.1, np.inf, np.nan])
def test_target_generation_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        build_target(_price_frame([100.0, 101.0]), threshold=threshold)


def test_prepare_dataset_aligns_features_and_target() -> None:
    data = _volatile_frame(rows=120)
    features, target = prepare_dataset(data)

    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    pd.testing.assert_index_equal(features.index, target.index)
    assert target.notna().all()
    assert set(target.unique()) == {0, 1}
    assert features.index[-1] == data.index[-2]


def test_feature_creation_includes_all_price_and_technical_features() -> None:
    data = _volatile_frame(rows=120)
    features = build_feature_frame(data)
    expected = set().union(*(set(columns) for columns in FEATURE_GROUPS.values()))

    assert expected <= set(features.columns)
    assert not {"Open", "High", "Low", "Close", "Volume"} & set(features.columns)
    assert features.index.equals(data.index)


def test_feature_creation_supports_custom_windows() -> None:
    data = _volatile_frame(rows=100)
    features = build_feature_frame(data, windows=[10, 30])

    assert {"MA_10", "MA_30", "rolling_return_10", "rolling_return_30"} <= set(
        features.columns
    )
    assert "MA_5" not in features.columns
    assert "MA_50" not in features.columns


def test_feature_creation_degrades_gracefully_without_ohlcv_columns() -> None:
    data = _price_frame([100.0, 101.0, 99.0, 102.0] * 20)
    features = build_feature_frame(data)

    assert {"bollinger_position", "bollinger_width", "atr_percent", "adx"} <= set(
        features.columns
    )
    assert {"obv", "volume_change", "volume_zscore_20"} <= set(features.columns)
    assert features[["bollinger_position", "atr_percent", "adx"]].isna().all().all()


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"Close": [100.0, 101.0]}),
        _price_frame([101.0, 100.0]).sort_index(ascending=False),
    ],
)
def test_feature_creation_validates_index(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_feature_frame(data)


def test_feature_creation_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError, match="windows"):
        build_feature_frame(_volatile_frame(rows=60), windows=[0, 5])


def test_context_features_are_lagged_before_alignment() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    context_close = pd.Series(np.arange(100.0, 220.0), index=index)
    context = pd.DataFrame(
        {
            "Close": context_close,
            "Volatility": np.linspace(0.1, 0.2, len(index)),
        },
        index=index,
    )
    features = build_feature_frame(
        _volatile_frame(rows=120),
        context_frames={"INDEX": context},
        context_lag=2,
    )

    row = 20
    expected_return = context_close.iloc[row - 2] / context_close.iloc[row - 3] - 1
    assert features.loc[features.index[row], "INDEX_return_1"] == pytest.approx(
        expected_return
    )
    assert features.loc[features.index[row], "INDEX_volatility_indicator"] == pytest.approx(
        context["Volatility"].iloc[row - 2]
    )


def test_context_features_never_use_future_context_values() -> None:
    data = make_price_frame(rows=150)
    context = make_price_frame(rows=150, start=200.0)
    features = build_feature_frame(data, windows=[5, 20], context_frames={"INDEX": context})

    modified = context.copy()
    modified.iloc[100, modified.columns.get_loc("Close")] *= 2.0
    modified_features = build_feature_frame(
        data,
        windows=[5, 20],
        context_frames={"INDEX": modified},
    )

    context_columns = [
        column for column in features.columns if column.startswith("INDEX_")
    ]
    pd.testing.assert_frame_equal(
        features.iloc[:101][context_columns],
        modified_features.iloc[:101][context_columns],
    )


def test_context_lag_must_be_positive() -> None:
    with pytest.raises(ValueError, match="context_lag"):
        build_feature_frame(
            _volatile_frame(rows=60),
            context_frames={"INDEX": _volatile_frame(rows=60, seed=2)},
            context_lag=0,
    )


def test_target_and_features_are_constructed_separately() -> None:
    data = make_price_frame(rows=120)
    features = build_feature_frame(data, windows=[5, 20])
    target = build_target(data)

    modified = data.copy()
    modified.iloc[80, modified.columns.get_loc("Close")] *= 1.5
    modified_features = build_feature_frame(modified, windows=[5, 20])
    modified_target = build_target(modified)

    pd.testing.assert_frame_equal(features.iloc[:80], modified_features.iloc[:80])
    pd.testing.assert_series_equal(target.iloc[:79], modified_target.iloc[:79])
    assert target.iloc[79] != modified_target.iloc[79] or target.iloc[80] != modified_target.iloc[80]


def test_rsi_does_not_use_future_prices() -> None:
    data = make_price_frame(rows=120)
    features_full = build_feature_frame(data, windows=[5])
    modified = data.copy()
    modified.iloc[80, modified.columns.get_loc("Close")] *= 1.5
    features_modified = build_feature_frame(modified, windows=[5])

    for row in range(20, 80):
        diff = abs(features_full["rsi_14"].iloc[row] - features_modified["rsi_14"].iloc[row])
        assert diff < 1e-9


def test_moving_average_does_not_use_future_prices() -> None:
    data = make_price_frame(rows=120)
    features_full = build_feature_frame(data, windows=[20])
    modified = data.copy()
    modified.iloc[90, modified.columns.get_loc("Close")] *= 2.0
    features_modified = build_feature_frame(modified, windows=[20])

    for row in range(20, 90):
        diff = abs(features_full["MA_20"].iloc[row] - features_modified["MA_20"].iloc[row])
        assert diff < 1e-9


def test_macd_does_not_use_future_prices() -> None:
    data = make_price_frame(rows=150)
    features_full = build_feature_frame(data)
    modified = data.copy()
    modified.iloc[100, modified.columns.get_loc("Close")] *= 1.3
    features_modified = build_feature_frame(modified)

    for row in range(90):
        diff = abs(features_full["macd"].iloc[row] - features_modified["macd"].iloc[row])
        assert diff < 1e-9


def test_return_feature_uses_only_current_and_prior_close() -> None:
    data = make_price_frame(rows=50)
    features = build_feature_frame(data)

    for row in range(1, len(data)):
        expected = data["Close"].iloc[row] / data["Close"].iloc[row - 1] - 1
        diff = abs(features["return_1"].iloc[row] - expected)
        assert diff < 1e-9


def test_feature_frame_is_deterministic_and_does_not_mutate_input() -> None:
    data = _volatile_frame(rows=100)
    snapshot = data.copy()
    first = build_feature_frame(data, windows=[5, 20])
    second = build_feature_frame(data, windows=[5, 20])

    pd.testing.assert_frame_equal(data, snapshot)
    pd.testing.assert_frame_equal(first, second)


def test_feature_preprocessor_fit_transform_and_metadata() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {"a": [0.0, 2.0, np.nan, 100.0], "b": [1.0, 1.0, 3.0, 5.0]},
        index=index,
    )
    preprocessor = FeaturePreprocessor()
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape == features.shape
    assert transformed.index.equals(features.index)
    assert preprocessor.get_feature_names_out().tolist() == ["a", "b"]
    assert preprocessor.metadata() == {
        "feature_names": ["a", "b"],
        "imputer_strategy": "median",
        "scaler": "StandardScaler",
    }
    assert np.isfinite(transformed.to_numpy()).all()
    assert transformed.mean().abs().max() < 1e-12


def test_feature_preprocessor_fits_only_training_statistics() -> None:
    train = pd.DataFrame({"a": [0.0, 2.0, 100.0], "b": [1.0, 3.0, 5.0]})
    test = pd.DataFrame({"a": [1000.0], "b": [1000.0]})
    preprocessor = FeaturePreprocessor().fit(train)
    transformed = preprocessor.transform(test)

    assert preprocessor.imputer.statistics_[0] == pytest.approx(2.0)
    assert preprocessor.scaler.mean_[0] == pytest.approx(34.0)
    assert transformed.iloc[0, 0] > 10.0


def test_feature_preprocessor_reindexes_missing_and_extra_columns() -> None:
    train = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [1.0, 3.0, 5.0]})
    test = pd.DataFrame({"b": [2.0], "extra": [99.0]})
    preprocessor = FeaturePreprocessor(feature_names=["a", "b"]).fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.columns.tolist() == ["a", "b"]
    assert np.isfinite(transformed.to_numpy()).all()


def test_feature_preprocessor_rejects_unfitted_empty_and_missing_columns() -> None:
    features = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not been fitted"):
        FeaturePreprocessor().transform(features)
    with pytest.raises(ValueError, match="empty"):
        FeaturePreprocessor().fit(features.iloc[0:0])
    with pytest.raises(KeyError, match="Missing feature"):
        FeaturePreprocessor(feature_names=["missing"]).fit(features)


def test_time_aware_split_is_chronological_with_purge_gap() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    features = pd.DataFrame({"x": np.arange(100)}, index=index)
    target = pd.Series(np.arange(100), index=index)
    train_x, val_x, test_x, train_y, val_y, test_y = time_aware_split(
        features,
        target,
    )

    assert (len(train_x), len(val_x), len(test_x)) == (60, 20, 18)
    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)
    assert len(test_x) == len(test_y)
    assert train_x.index[-1] < val_x.index[0] < test_x.index[0]
    assert (val_x.index[0] - train_x.index[-1]).days == 2
    assert (test_x.index[0] - val_x.index[-1]).days == 2
    assert train_x.index.union(val_x.index).union(test_x.index).is_unique


def test_time_aware_split_without_target_returns_three_frames() -> None:
    features = pd.DataFrame(
        {"x": np.arange(100)},
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )
    train, validation, test = time_aware_split(features)

    assert (len(train), len(validation), len(test)) == (60, 20, 18)
    assert train.index[-1] < validation.index[0] < test.index[0]


def test_time_aware_split_rejects_invalid_fractions_and_small_data() -> None:
    features = pd.DataFrame(
        {"x": np.arange(20)},
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )
    target = pd.Series(np.arange(20), index=features.index)
    invalid = [
        (0.0, 0.5, 0.5),
        (0.5, -0.1, 0.6),
        (0.5, 0.2, 0.4),
    ]
    for fractions in invalid:
        with pytest.raises(ValueError):
            time_aware_split(
                features,
                target,
                train_fraction=fractions[0],
                validation_fraction=fractions[1],
                test_fraction=fractions[2],
            )
    with pytest.raises(ValueError, match="not enough rows"):
        time_aware_split(features, target, min_train_rows=30)


def test_time_aware_split_rejects_partial_index_overlap() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    features = pd.DataFrame({"x": np.arange(10)}, index=index)
    target = pd.Series(np.arange(8), index=index[:8])

    with pytest.raises(ValueError, match="overlap"):
        align_features_target(features, target)


def test_backward_compatible_train_test_split_is_chronological() -> None:
    frame = pd.DataFrame(
        {"x": np.arange(100)},
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )
    train, test = train_test_split(frame, test_fraction=0.2)

    assert (len(train), len(test)) == (80, 20)
    assert train.index[-1] < test.index[0]
    pd.testing.assert_frame_equal(pd.concat([train, test]), frame)
    with pytest.raises(ValueError):
        train_test_split(frame, test_fraction=0.0)


def test_train_pipeline_trains_and_compares_all_models(
    trained_result: TrainingResult,
) -> None:
    assert set(trained_result.bundles) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert set(trained_result.comparisons) == set(trained_result.bundles)
    assert sum(comparison.selected for comparison in trained_result.comparisons.values()) == 1


def test_train_pipeline_fits_preprocessors_and_estimators(
    trained_result: TrainingResult,
) -> None:
    for bundle in trained_result.bundles.values():
        assert bundle.preprocessor.fitted_feature_names_
        assert hasattr(bundle.estimator, "classes_")
        assert len(bundle.metadata.feature_names) == len(bundle.preprocessor.fitted_feature_names_)


def test_model_selection_uses_validation_not_test_performance(
    trained_result: TrainingResult,
) -> None:
    best_validation = max(
        trained_result.comparisons,
        key=lambda name: trained_result.comparisons[name].validation_metrics[
            "balanced_accuracy"
        ],
    )
    assert trained_result.selected_model_name == best_validation
    selected = trained_result.comparisons[best_validation]
    assert selected.selected
    assert all(
        comparison.selected == (name == best_validation)
        for name, comparison in trained_result.comparisons.items()
    )


def test_training_metrics_include_classification_calibration_and_baseline(
    trained_result: TrainingResult,
) -> None:
    metrics = trained_result.comparisons["logistic_regression"].validation_metrics
    required = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "log_loss",
        "brier_score",
        "calibration",
        "balanced_accuracy",
        "baseline",
    }
    assert required <= set(metrics)
    assert len(metrics["confusion_matrix"]) == 2
    assert metrics["calibration"]["n_bins"] == 10
    assert "balanced_accuracy" in metrics["baseline"]
    assert "roc_auc" in metrics["baseline"]


def test_model_metadata_contains_version_window_libraries_and_disclaimer(
    trained_result: TrainingResult,
) -> None:
    metadata = trained_result.selected_bundle.metadata
    assert metadata.schema_version == "1.0"
    assert metadata.model_version.startswith(f"{metadata.model_name}-v1-")
    assert len(metadata.model_version) > 20
    assert metadata.training_start <= metadata.training_end
    assert metadata.train_rows > 0
    assert metadata.validation_rows > 0
    assert metadata.test_rows > 0
    assert metadata.libraries["python"]
    assert metadata.libraries["scikit-learn"] != "unavailable"
    assert metadata.libraries["xgboost"] != "unavailable"
    assert "Educational" in metadata.disclaimer
    assert "market_context" not in metadata.feature_groups


def test_model_version_is_deterministic_and_configuration_sensitive() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    feature_names = ["return_1", "volatility_20"]
    config = TrainingConfig(random_state=42, n_estimators=15)

    first = _version_for("logistic_regression", feature_names, config, index)
    repeated = _version_for("logistic_regression", feature_names, config, index)
    changed = _version_for(
        "logistic_regression",
        feature_names,
        TrainingConfig(random_state=42, n_estimators=16),
        index,
    )

    assert first == repeated
    assert first != changed
    assert first.startswith("logistic_regression-v1-")


def test_train_model_fits_default_logistic_regression(
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = aligned_dataset
    mask = ~features.isna().any(axis=1)
    model = train_model(features[mask], target[mask])

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert model.coef_.shape[1] == features[mask].shape[1]


def test_train_model_preserves_objects_without_fit_method() -> None:
    features = make_price_frame(rows=50)
    target = pd.Series(np.zeros(50), index=features.index)
    sentinel = object()

    assert train_model(features, target, model=sentinel) is sentinel


def test_train_model_returns_none_for_empty_or_non_overlapping_input() -> None:
    assert train_model(pd.DataFrame(), pd.Series(dtype=float)) is None
    assert train_model(None, pd.Series(dtype=float)) is None
    assert train_model(pd.DataFrame(), None) is None
    features = make_price_frame(rows=50)
    target = pd.Series(np.zeros(30), index=features.index[:30])
    assert train_model(features, target) is None


def test_train_model_rejects_single_class_target(
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = aligned_dataset
    mask = ~features.isna().any(axis=1)
    constant_target = pd.Series(1, index=target[mask].index, dtype=int)

    with pytest.raises(ValueError, match="2 classes"):
        train_model(features[mask], constant_target)


def test_train_models_rejects_single_class_target(
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = aligned_dataset
    mask = ~features.isna().any(axis=1)
    constant_target = pd.Series(1, index=target[mask].index, dtype=int)

    with pytest.raises(ValueError, match="both direction classes"):
        train_models(features[mask], constant_target, config=TrainingConfig(n_estimators=5))


def test_prediction_result_contains_probabilities_metadata_and_disclaimer(
    trained_result: TrainingResult,
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = aligned_dataset
    result = predict(trained_result.selected_bundle, features, symbol="TEST")

    assert isinstance(result, PredictionResult)
    assert result.symbol == "TEST"
    assert result.prediction in {"up", "down"}
    assert result.is_available
    assert result.probability == pytest.approx(result.confidence)
    assert 0.5 <= result.probability <= 1.0
    assert 0.0 <= result.up_probability <= 1.0
    assert 0.0 <= result.down_probability <= 1.0
    assert result.up_probability + result.down_probability == pytest.approx(1.0)
    assert result.model_name == trained_result.selected_bundle.metadata.model_name
    assert result.model_version == trained_result.selected_bundle.metadata.model_version
    assert result.as_of == features.index[-1].isoformat()
    datetime.fromisoformat(result.timestamp)
    assert "Educational prediction" in result.disclaimer
    assert "Binary direction" in result.target_definition


def test_prediction_result_explanation_is_populated_and_labeled(
    trained_result: TrainingResult,
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = aligned_dataset
    result = predict(trained_result.selected_bundle, features)

    assert result.explanation
    assert len(result.explanation) <= 10
    assert {"feature", "absolute_contribution", "scope", "interpretation"} <= set(
        result.explanation[0]
    )
    assert result.explanation[0]["scope"] == "local_linear"
    assert result.explanation[0]["interpretation"] == "association, not causation"


def test_predict_many_returns_one_result_per_row(
    trained_result: TrainingResult,
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = aligned_dataset
    results = predict_many(trained_result.selected_bundle, features.iloc[-3:], symbol="TEST")

    assert len(results) == 3
    assert all(isinstance(result, PredictionResult) for result in results)
    assert all(result.symbol == "TEST" for result in results)


def test_predict_from_history_builds_latest_features(
    trained_result: TrainingResult,
) -> None:
    data = _volatile_frame(rows=200)
    result = predict_from_history(trained_result.selected_bundle, data, symbol="TEST")

    assert result.is_available
    assert result.prediction in {"up", "down"}
    assert result.model_version is not None
    assert result.as_of is not None
    assert result.explanation


def test_predict_accepts_saved_bundle_path(
    trained_result: TrainingResult,
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = aligned_dataset
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        save_model_bundle(trained_result.selected_bundle, path)
        result = predict(str(path), features, symbol="TEST")

    assert result.is_available
    assert result.model_version == trained_result.selected_bundle.metadata.model_version


def test_predict_with_bare_estimator_and_preprocessor(
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = aligned_dataset
    preprocessor = FeaturePreprocessor().fit(features.iloc[:140])
    estimator = LogisticRegression(max_iter=500, random_state=42).fit(
        preprocessor.transform(features.iloc[:140]),
        target.iloc[:140],
    )
    result = predict(estimator, features.iloc[[-1]], preprocessor=preprocessor)

    assert result.is_available
    assert result.model_version is None
    assert result.model_name is None
    assert result.explanation


def test_predict_with_bare_estimator_without_preprocessor() -> None:
    estimator = LogisticRegression().fit(
        np.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.1], [0.0, 0.1]]),
        np.array([0, 0, 1, 1]),
    )
    features = pd.DataFrame(
        {"a": [0.1], "b": [0.0]},
        index=pd.DatetimeIndex(["2024-01-01"]),
    )
    result = predict(estimator, features)

    assert result.is_available
    assert result.explanation


def test_predict_returns_unavailable_for_invalid_or_empty_input(
    trained_result: TrainingResult,
) -> None:
    bundle = trained_result.selected_bundle
    unavailable = [
        predict(None, pd.DataFrame()),
        predict(object(), None),
        predict(object(), pd.DataFrame()),
        predict(object(), pd.DataFrame({"a": [1.0]})),
    ]

    assert all(result.prediction == "unavailable" for result in unavailable)
    assert all(not result.is_available for result in unavailable)
    assert all(result.probability is None for result in unavailable)
    assert predict(bundle, pd.DataFrame()).as_of is None


def test_prediction_to_dict_returns_json_safe_mapping(
    trained_result: TrainingResult,
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = aligned_dataset
    result = predict(trained_result.selected_bundle, features)
    payload = prediction_to_dict(result)

    assert payload["prediction"] == result.prediction
    assert len(payload["explanation"]) == len(result.explanation)
    json.dumps(payload)


def test_classification_metrics_return_known_values_and_baseline() -> None:
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    metrics = classification_metrics(y_true, y_pred)

    expected = 4 / 6
    assert metrics["accuracy"] == pytest.approx(expected)
    assert metrics["precision"] == pytest.approx(expected)
    assert metrics["recall"] == pytest.approx(expected)
    assert metrics["f1"] == pytest.approx(expected)
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
    assert metrics["roc_auc"] is None
    assert metrics["log_loss"] is None
    assert metrics["brier_score"] is None
    assert metrics["calibration"] is None
    assert metrics["baseline"]["majority_class"] == 0
    assert metrics["baseline"]["accuracy"] == pytest.approx(expected)
    assert metrics["baseline"]["balanced_accuracy"] == 0.5
    assert metrics["baseline"]["roc_auc"] is None


def test_classification_metrics_compute_probability_metrics() -> None:
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.35, 0.8]
    metrics = classification_metrics(y_true, y_true, y_proba)

    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["log_loss"] >= 0.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["calibration"]["expected_calibration_error"] >= 0.0
    assert len(metrics["calibration"]["reliability"]) == 10


def test_classification_metrics_handle_single_class_without_fabricated_auc() -> None:
    metrics = classification_metrics([1, 1, 1], [1, 1, 1], [0.8, 0.9, 0.7])

    assert metrics["roc_auc"] is None
    assert metrics["calibration"] is not None
    assert metrics["baseline"]["majority_class"] == 1


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_proba"),
    [
        ([], [], None),
        ([0, 1], [0], None),
        ([0, 1], [0, 1], [0.2]),
        ([0, 1], [0, 1], [1.2, 0.2]),
    ],
)
def test_classification_metrics_reject_invalid_inputs(
    y_true: list[int],
    y_pred: list[int],
    y_proba: list[float] | None,
) -> None:
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_pred, y_proba)


def test_calibration_metrics_perfect_predictions_have_zero_ece() -> None:
    calibration = calibration_metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], n_bins=4)

    assert calibration["expected_calibration_error"] == 0.0
    assert calibration["n_bins"] == 4
    assert len(calibration["reliability"]) == 4


def test_compare_model_metrics_selects_best_validation_score() -> None:
    results = {
        "a": {"balanced_accuracy": 0.55, "f1": 0.4},
        "b": {"balanced_accuracy": 0.7, "f1": 0.6},
    }
    selection = compare_model_metrics(results)

    assert selection["selected_model"] == "b"
    assert selection["primary_metric"] == "balanced_accuracy"
    assert selection["validation_scores"] == {
        "a": 0.55,
        "b": 0.7,
    }


def test_compare_model_metrics_falls_back_to_f1_and_rejects_empty() -> None:
    assert compare_model_metrics({"a": {"f1": 0.6}})["selected_model"] == "a"
    with pytest.raises(ValueError, match="at least one"):
        compare_model_metrics({})
    with pytest.raises(ValueError, match="selection metric"):
        compare_model_metrics({"a": {"accuracy": 0.6}})


def test_evaluate_classifier_uses_fitted_estimator() -> None:
    estimator = LogisticRegression().fit(
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 0, 1, 1]),
    )
    features = pd.DataFrame({"x": [0.5, 2.5]}, index=pd.date_range("2024-01-01", periods=2))
    target = pd.Series([0, 1], index=features.index)
    metrics = evaluate_classifier(estimator, features, target)

    assert set(metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_regression_metrics_known_values_and_edge_cases() -> None:
    assert regression_metrics([1, 2, 3], [1, 2, 3]) == {"mse": 0.0, "mae": 0.0, "r2": 1.0}
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])
    assert metrics["mse"] == pytest.approx(1 / 3)
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["r2"] < 1.0
    assert regression_metrics([1, 2, 3], [10, 10, 10])["r2"] < 0.0
    assert regression_metrics([], []) == {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    assert regression_metrics([5], [5])["r2"] == 0.0


def test_feature_importance_orders_by_absolute_magnitude() -> None:
    ordered = feature_importance(["a", "b", "c"], [0.1, -0.5, 0.2], top_n=2)

    assert [item["feature"] for item in ordered] == ["b", "c"]
    assert ordered[0]["importance"] == -0.5
    with pytest.raises(ValueError, match="same length"):
        feature_importance(["a"], [0.1, 0.2])


def test_explain_prediction_distinguishes_local_and_global_scope(
    trained_result: TrainingResult,
) -> None:
    linear = LogisticRegression().fit(
        np.array([[-1.0, 0.0], [1.0, 0.1], [0.0, -1.0], [0.1, 1.0]]),
        np.array([0, 1, 0, 1]),
    )
    linear_explanation = explain_prediction(linear, ["a", "b"], pd.Series([1.0, -1.0]))
    assert linear_explanation[0]["scope"] == "local_linear"

    data = _volatile_frame(rows=200)
    row = trained_result.selected_bundle.preprocessor.transform(
        build_feature_frame(data)
    ).iloc[[0]]
    tree_explanation = explain_prediction(
        trained_result.bundles["random_forest"].estimator,
        trained_result.selected_bundle.feature_names,
        row,
    )
    assert tree_explanation[0]["scope"] == "global_tree"
    assert "not a local contribution" in tree_explanation[0]["interpretation"]


def test_explain_model_returns_global_scores() -> None:
    estimator = LogisticRegression().fit(
        np.array([[-1.0], [1.0], [2.0], [-2.0]]),
        np.array([0, 1, 1, 0]),
    )
    explanations = explain_model(estimator, ["x"], top_n=1)

    assert len(explanations) == 1
    assert explanations[0]["feature"] == "x"


def test_model_bundle_persistence_roundtrip_and_metadata_sidecar(
    trained_result: TrainingResult,
) -> None:
    bundle = trained_result.selected_bundle
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        saved = save_model_bundle(bundle, path)
        metadata_path = Path(str(saved) + ".json")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded = load_model_bundle(saved)

        assert saved.exists()
        assert metadata_path.exists()
        assert payload["model_version"] == bundle.metadata.model_version
        assert payload["feature_names"] == list(bundle.metadata.feature_names)
        assert loaded.metadata == bundle.metadata
        assert loaded.preprocessor.fitted_feature_names_ == bundle.preprocessor.fitted_feature_names_
        assert type(loaded.estimator) is type(bundle.estimator)
        assert not path.with_name(f".{path.name}.tmp").exists()


def test_model_bundle_persistence_supports_inference_after_reload(
    trained_result: TrainingResult,
    aligned_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = aligned_dataset
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        save_model_bundle(trained_result.selected_bundle, path)
        loaded = load_model_bundle(path)
        result = predict(loaded, features, symbol="TEST")

    assert result.is_available
    assert result.model_version == trained_result.selected_bundle.metadata.model_version
    assert result.explanation


def test_model_bundle_loader_rejects_wrong_type_and_schema(
    trained_result: TrainingResult,
) -> None:
    bundle = trained_result.selected_bundle
    invalid_metadata = replace(bundle.metadata, schema_version="99")
    invalid_bundle = ModelBundle(bundle.estimator, bundle.preprocessor, invalid_metadata)

    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        wrong_type = directory_path / "wrong.joblib"
        joblib.dump({"not": "a bundle"}, wrong_type)
        wrong_schema = directory_path / "schema.joblib"
        save_model_bundle(invalid_bundle, wrong_schema)

        with pytest.raises(TypeError, match="ModelBundle"):
            load_model_bundle(wrong_type)
        with pytest.raises(ValueError, match="unsupported model schema"):
            load_model_bundle(wrong_schema)


def test_model_metadata_is_json_safe() -> None:
    metadata = ModelMetadata(model_name="test", model_version="test-v1")
    json.dumps(metadata.to_dict())
    assert metadata.to_dict()["schema_version"] == "1.0"
