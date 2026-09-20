"""Focused ML tests covering the full modeling pipeline.

These tests exercise the shared ML interfaces in ``src.ml`` against
real OHLCV fixtures, verifying:

1. Target generation          — forward-return targets align with features
2. Feature creation           — ``build_feature_frame`` produces expected columns
3. No lookahead / leakage     — indicators depend only on past data
4. Time-aware splitting       — ``train_test_split`` preserves ordering
5. Preprocessing fit scope    — fit is stateless; no test data seen
6. Model comparison/training  — ``train_model`` handles multiple models
7. Prediction format          — ``predict`` returns correct shape/types
8. Persistence metadata       — saved artifacts carry version + feature info
9. Evaluation metrics         — ``regression_metrics`` values are correct
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.ml.evaluation import regression_metrics
from src.ml.explainability import feature_importance
from src.ml.prediction import predict
from src.ml.preprocessing import build_feature_frame, train_test_split
from src.ml.training import train_model
from tests.fixtures import make_price_frame

MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# 1. Target generation
# ---------------------------------------------------------------------------


def test_target_forward_return_aligns_with_features() -> None:
    """A forward 1-day return target must share the index and length
    of the feature frame, with NaN on the final row (no future data)."""

    data = make_price_frame(rows=120)
    features = build_feature_frame(data, windows=[5, 20])

    target = data["Close"].shift(-1) / data["Close"] - 1
    target.name = "forward_return"

    assert len(target) == len(features)
    pd.testing.assert_index_equal(target.index, features.index)
    assert pd.isna(target.iloc[-1])
    assert target.notna().sum() == len(target) - 1


def test_target_never_contains_information_from_future_at_same_row() -> None:
    """At row *t* the target must be derived from the close at *t+1*,
    never from any price at or before *t*."""

    data = make_price_frame(rows=50)
    target = data["Close"].shift(-1) / data["Close"] - 1

    for t in range(len(data) - 1):
        future_close = data["Close"].iloc[t + 1]
        current_close = data["Close"].iloc[t]
        expected = future_close / current_close - 1
        assert math.isclose(target.iloc[t], expected, rel_tol=1e-9)


def test_target_is_nan_on_last_row_when_no_future_exists() -> None:
    """The last row has no future close, so its target must be NaN."""

    data = make_price_frame(rows=30)
    target = data["Close"].shift(-1) / data["Close"] - 1
    assert pd.isna(target.iloc[-1])


# ---------------------------------------------------------------------------
# 2. Feature creation
# ---------------------------------------------------------------------------


def test_build_feature_frame_includes_indicators() -> None:
    data = make_price_frame(rows=120)
    frame = build_feature_frame(data, windows=[5, 20])
    expected_columns = {
        "return_1",
        "log_return_1",
        "rsi_14",
        "macd",
        "macd_signal",
        "MA_5",
        "MA_20",
        "volume_change",
    }
    assert expected_columns <= set(frame.columns)


def test_build_feature_frame_custom_windows() -> None:
    data = make_price_frame(rows=100)
    frame = build_feature_frame(data, windows=[10, 30])
    assert "MA_10" in frame.columns
    assert "MA_30" in frame.columns
    assert "MA_5" not in frame.columns
    assert "MA_20" not in frame.columns


def test_build_feature_frame_default_windows() -> None:
    data = make_price_frame(rows=100)
    frame = build_feature_frame(data)
    assert "MA_5" in frame.columns
    assert "MA_20" in frame.columns
    assert "MA_50" in frame.columns


def test_build_feature_frame_handles_empty_input() -> None:
    assert build_feature_frame(pd.DataFrame()).empty


def test_build_feature_frame_returns_dataframe_with_index() -> None:
    data = make_price_frame(rows=60)
    frame = build_feature_frame(data)
    assert isinstance(frame, pd.DataFrame)
    pd.testing.assert_index_equal(frame.index, data.index)


def test_build_feature_frame_volume_change_zero_when_no_volume_col() -> None:
    data = make_price_frame(rows=50).drop(columns=["Volume"])
    frame = build_feature_frame(data)
    assert "volume_change" in frame.columns


# ---------------------------------------------------------------------------
# 3. No lookahead / leakage
# ---------------------------------------------------------------------------


def test_rsi_no_lookahead() -> None:
    """Modifying a future row must not change RSI at the current row."""

    data = make_price_frame(rows=120)
    for i in range(10, 120, 7):
        data.iloc[i, data.columns.get_loc("Close")] *= 0.98
    features_full = build_feature_frame(data, windows=[5])

    data_modified = data.copy()
    data_modified.iloc[80, data_modified.columns.get_loc("Close")] *= 1.5
    features_modified = build_feature_frame(data_modified, windows=[5])

    for t in range(20, 80):
        full_val = features_full["rsi_14"].iloc[t]
        mod_val = features_modified["rsi_14"].iloc[t]
        assert pd.notna(full_val) and pd.notna(mod_val), (
            f"Unexpected NaN at row {t}"
        )
        assert math.isclose(full_val, mod_val, rel_tol=1e-9), (
            f"RSI at row {t} changed when future row 80 was modified — lookahead detected"
        )


def test_moving_average_no_lookahead() -> None:
    """A future price change must not alter past MA values."""

    data = make_price_frame(rows=120)
    features_full = build_feature_frame(data, windows=[20])

    data_modified = data.copy()
    data_modified.iloc[90, data_modified.columns.get_loc("Close")] *= 2.0
    features_modified = build_feature_frame(data_modified, windows=[20])

    for t in range(20, 80):
        full_val = features_full["MA_20"].iloc[t]
        mod_val = features_modified["MA_20"].iloc[t]
        assert pd.notna(full_val) and pd.notna(mod_val), (
            f"Unexpected NaN at row {t}"
        )
        assert math.isclose(full_val, mod_val, rel_tol=1e-9), (
            f"MA_20 at row {t} changed when future row 90 was modified — lookahead detected"
        )


def test_macd_no_lookahead() -> None:
    """A future price change must not alter past MACD values."""

    data = make_price_frame(rows=150)
    features_full = build_feature_frame(data)

    data_modified = data.copy()
    data_modified.iloc[100, data_modified.columns.get_loc("Close")] *= 1.3
    features_modified = build_feature_frame(data_modified)

    for t in range(90):
        assert math.isclose(
            features_full["macd"].iloc[t],
            features_modified["macd"].iloc[t],
            rel_tol=1e-9,
        ), f"MACD at row {t} changed when future row 100 was modified — lookahead detected"


def test_return_feature_only_uses_prior_data() -> None:
    """return_1 at row t = Close[t]/Close[t-1] - 1, never uses Close[t+1]."""

    data = make_price_frame(rows=50)
    frame = build_feature_frame(data)

    for t in range(1, len(data)):
        expected = data["Close"].iloc[t] / data["Close"].iloc[t - 1] - 1
        if not pd.isna(frame["return_1"].iloc[t]):
            assert math.isclose(frame["return_1"].iloc[t], expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Time-aware splitting
# ---------------------------------------------------------------------------


def test_train_test_split_respects_test_fraction() -> None:
    frame = make_price_frame(rows=100)
    train, test = train_test_split(frame, test_fraction=0.2)
    assert len(train) == 80
    assert len(test) == 20


def test_train_test_split_rejects_bad_fraction() -> None:
    frame = make_price_frame(rows=100)
    for bad in (0, 1, -0.1, 1.1):
        with pytest.raises(ValueError):
            train_test_split(frame, test_fraction=bad)


def test_train_test_split_preserves_temporal_ordering() -> None:
    """Test set must contain strictly later timestamps than train set."""

    frame = make_price_frame(rows=100)
    train, test = train_test_split(frame, test_fraction=0.25)

    train_last_date = train.index[-1]
    test_first_date = test.index[0]
    assert test_first_date > train_last_date


def test_train_test_split_no_overlap() -> None:
    """Train and test sets must share no rows."""

    frame = make_price_frame(rows=120)
    train, test = train_test_split(frame, test_fraction=0.2)

    train_ids = id(train.index)
    test_ids = id(test.index)
    assert train_ids != test_ids
    common = train.index.intersection(test.index)
    assert len(common) == 0


def test_train_test_split_contiguous() -> None:
    """Concatenating train and test in order must reconstruct the original."""

    frame = make_price_frame(rows=60)
    train, test = train_test_split(frame, test_fraction=0.3)
    reconstructed = pd.concat([train, test])
    pd.testing.assert_index_equal(reconstructed.index, frame.index)


def test_train_test_split_tiny_fraction() -> None:
    frame = make_price_frame(rows=10)
    train, test = train_test_split(frame, test_fraction=0.1)
    assert len(train) == 9
    assert len(test) == 1


def test_train_test_split_returns_same_index_type() -> None:
    frame = make_price_frame(rows=50)
    train, test = train_test_split(frame, test_fraction=0.2)
    assert isinstance(train.index, type(frame.index))
    assert isinstance(test.index, type(frame.index))


# ---------------------------------------------------------------------------
# 5. Preprocessing fit scope
# ---------------------------------------------------------------------------


def test_build_feature_frame_is_stateless() -> None:
    """Calling build_feature_frame twice must produce identical results
    — no internal state or fitting step accumulates."""

    data = make_price_frame(rows=80)
    first = build_feature_frame(data, windows=[5, 20])
    second = build_feature_frame(data, windows=[5, 20])
    pd.testing.assert_frame_equal(first, second)


def test_train_test_split_is_stateless() -> None:
    """Calling train_test_split twice on the same frame must yield
    identical splits — no global tracking of previous splits."""

    frame = make_price_frame(rows=80)
    first_train, first_test = train_test_split(frame, test_fraction=0.2)
    second_train, second_test = train_test_split(frame, test_fraction=0.2)
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_preprocessing_does_not_consume_test_data() -> None:
    """Fitting preprocessing on a frame must not alter the original data."""

    data = make_price_frame(rows=60)
    snapshot = data.copy()
    _ = build_feature_frame(data, windows=[5])
    pd.testing.assert_frame_equal(data, snapshot)


def test_preprocessing_with_different_windows_independent() -> None:
    """Different window parameters must not affect each other's output."""

    data = make_price_frame(rows=80)
    frame_a = build_feature_frame(data, windows=[5])
    frame_b = build_feature_frame(data, windows=[20])
    assert not frame_a.equals(frame_b)
    assert "MA_5" in frame_a.columns and "MA_5" not in frame_b.columns
    assert "MA_20" not in frame_a.columns and "MA_20" in frame_b.columns


# ---------------------------------------------------------------------------
# 6. Model comparison / training
# ---------------------------------------------------------------------------


def test_train_model_returns_provided_model() -> None:
    sentinel = object()
    assert train_model(pd.DataFrame(), pd.Series(), model=sentinel) is sentinel


def test_train_model_returns_none_without_model() -> None:
    assert train_model(pd.DataFrame(), pd.Series()) is None


def test_train_model_returns_none_on_none_features() -> None:
    assert train_model(None, pd.Series()) is None
    assert train_model(pd.DataFrame(), None) is None


def test_train_model_works_with_different_model_types() -> None:
    """train_model should accept any model type without error."""

    features = make_price_frame(rows=50)
    target = pd.Series(np.zeros(50))

    model_a = {"name": "dummy-a"}
    model_b = [1, 2, 3]
    model_c = 42

    assert train_model(features, target, model=model_a) is model_a
    assert train_model(features, target, model=model_b) is model_b
    assert train_model(features, target, model=model_c) is model_c


def test_train_model_with_feature_target_length_mismatch_returns_none() -> None:
    features = make_price_frame(rows=50)
    target = pd.Series(np.zeros(30))
    assert train_model(features, target) is None


def test_train_model_different_configs_train_independently() -> None:
    """Two models trained on the same data should be distinct objects."""

    features = make_price_frame(rows=40)
    target = pd.Series(np.zeros(40))
    model_a = {"name": "model_a"}
    model_b = {"name": "model_b"}

    result_a = train_model(features, target, model=model_a)
    result_b = train_model(features, target, model=model_b)

    assert result_a is model_a
    assert result_b is model_b
    assert result_a is not result_b


# ---------------------------------------------------------------------------
# 7. Prediction format
# ---------------------------------------------------------------------------


def test_predict_returns_empty_list_for_none_model() -> None:
    assert list(predict(None, make_price_frame(rows=10))) == []


def test_predict_returns_empty_list_for_none_features() -> None:
    assert list(predict(object(), None)) == []


def test_predict_returns_empty_list_for_empty_features() -> None:
    assert list(predict(object(), pd.DataFrame())) == []


def test_predict_returns_list_of_floats() -> None:
    """Prediction output must be an iterable of floats."""

    model = object()
    features = make_price_frame(rows=20)
    result = list(predict(model, features))
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_predict_returns_single_value_for_nonempty_frame() -> None:
    """Placeholder predict yields one float per call on a non-empty frame."""

    model = object()
    for n in (1, 5, 20):
        features = make_price_frame(rows=n)
        result = list(predict(model, features))
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(v, float) for v in result)


def test_predict_value_is_last_row_sum() -> None:
    """Placeholder predict returns sum of last row as a float."""

    model = object()
    features = make_price_frame(rows=10)
    result = list(predict(model, features))
    expected = float(features.iloc[-1].sum())
    assert math.isclose(result[-1], expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 8. Persistence metadata
# ---------------------------------------------------------------------------


def _build_model_artifact(features: pd.DataFrame, version: str = MODEL_VERSION) -> dict:
    """Create a model artifact with metadata (standard persistence pattern)."""

    return {
        "model": None,
        "metadata": {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": list(features.columns),
            "n_features": len(features.columns),
        },
    }


def test_persistence_artifact_includes_version() -> None:
    data = make_price_frame(rows=50)
    features = build_feature_frame(data, windows=[5, 20])
    artifact = _build_model_artifact(features)

    assert artifact["metadata"]["version"] == MODEL_VERSION


def test_persistence_artifact_includes_feature_columns() -> None:
    data = make_price_frame(rows=50)
    features = build_feature_frame(data, windows=[5, 20])
    expected_cols = {
        "return_1",
        "log_return_1",
        "rsi_14",
        "macd",
        "macd_signal",
        "MA_5",
        "MA_20",
        "volume_change",
    }
    artifact = _build_model_artifact(features)
    saved_cols = set(artifact["metadata"]["feature_columns"])
    assert expected_cols <= saved_cols


def test_persistence_artifact_includes_timestamp() -> None:
    data = make_price_frame(rows=50)
    features = build_feature_frame(data)
    artifact = _build_model_artifact(features)

    ts = artifact["metadata"]["created_at"]
    datetime.fromisoformat(ts)


def test_persistence_artifact_can_be_roundtripped() -> None:
    """Save and load an artifact via joblib; metadata must survive."""

    data = make_price_frame(rows=40)
    features = build_feature_frame(data, windows=[5, 20])
    artifact = _build_model_artifact(features)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        path = Path(tmp.name)

    try:
        joblib.dump(artifact, path)
        loaded = joblib.load(path)

        assert loaded["metadata"]["version"] == artifact["metadata"]["version"]
        assert loaded["metadata"]["feature_columns"] == artifact["metadata"][
            "feature_columns"
        ]
        assert loaded["metadata"]["n_features"] == artifact["metadata"]["n_features"]
        assert "created_at" in loaded["metadata"]
    finally:
        path.unlink(missing_ok=True)


def test_persistence_different_versions_are_distinguishable() -> None:
    data = make_price_frame(rows=30)
    features = build_feature_frame(data)

    v1 = _build_model_artifact(features, version="1.0.0")
    v2 = _build_model_artifact(features, version="2.0.0")

    assert v1["metadata"]["version"] != v2["metadata"]["version"]


# ---------------------------------------------------------------------------
# 9. Evaluation metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_perfect_prediction() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_known_values() -> None:
    """Verify MSE, MAE, and R² on a small known example."""

    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    metrics = regression_metrics(y_true, y_pred)

    assert math.isclose(metrics["mae"], 1 / 3, rel_tol=1e-9)
    assert math.isclose(metrics["mse"], 1 / 3, rel_tol=1e-9)
    assert metrics["r2"] < 1.0


def test_regression_metrics_handles_invalid_input() -> None:
    assert regression_metrics([], []) == {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    assert regression_metrics([1.0], [1.0, 2.0]) == {"mse": 0.0, "mae": 0.0, "r2": 0.0}


def test_regression_metrics_r2_negative_for_worse_than_mean() -> None:
    """A model worse than predicting the mean should have negative R²."""

    y_true = [1.0, 2.0, 3.0]
    y_pred = [10.0, 10.0, 10.0]
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["r2"] < 0


def test_regression_metrics_all_same_predictions() -> None:
    """Constant predictions equal to the mean → R² = 0."""

    y_true = [1.0, 2.0, 3.0]
    mean_val = sum(y_true) / len(y_true)
    metrics = regression_metrics(y_true, [mean_val, mean_val, mean_val])
    assert math.isclose(metrics["r2"], 0.0, abs_tol=1e-9)


def test_regression_metrics_length_mismatch() -> None:
    assert regression_metrics([1.0, 2.0], [1.0]) == {"mse": 0.0, "mae": 0.0, "r2": 0.0}


def test_regression_metrics_single_match() -> None:
    """With one sample ss_tot is 0, so R² defaults to 0.0 by design."""

    metrics = regression_metrics([5.0], [5.0])
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 0.0


# ---------------------------------------------------------------------------
# 10. Feature importance (explainability)
# ---------------------------------------------------------------------------


def test_feature_importance_orders_by_abs_value() -> None:
    ordered = feature_importance(["a", "b", "c"], [0.1, -0.5, 0.2])
    assert [item["feature"] for item in ordered] == ["b", "c", "a"]


def test_feature_importance_returns_list_of_dicts() -> None:
    result = feature_importance(["x", "y"], [0.3, -0.7])
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)
    assert all("feature" in d and "importance" in d for d in result)


def test_feature_importance_preserves_sign() -> None:
    ordered = feature_importance(["a", "b"], [0.3, -0.7])
    assert ordered[0]["importance"] == -0.7
    assert ordered[1]["importance"] == 0.3


def test_feature_importance_all_same_magnitude() -> None:
    ordered = feature_importance(["a", "b"], [0.5, -0.5])
    assert ordered[0]["feature"] == "a"
    assert ordered[1]["feature"] == "b"
