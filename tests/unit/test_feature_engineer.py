"""Tests for the FeatureEngineer class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import FeatureEngineer
from tests.fixtures import make_constant_frame, make_price_frame


class TestFeatureEngineer:
    def test_transform_returns_dataframe(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_preserves_original_columns(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_transform_adds_feature_columns(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        expected_features = engineer.feature_names()
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_transform_length_matches_input(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert len(result) == len(data)

    def test_transform_index_matches_input(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        pd.testing.assert_index_equal(result.index, data.index)

    def test_transform_raises_on_empty_dataframe(self) -> None:
        engineer = FeatureEngineer()
        with pytest.raises(ValueError):
            engineer.transform(pd.DataFrame())

    def test_transform_constant_price(self) -> None:
        """All indicators should handle constant prices gracefully."""
        data = make_constant_frame(rows=100, price=100.0)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert not result.empty
        assert result["RSI"].dropna().eq(100.0).all()

    def test_transform_insufficient_history(self) -> None:
        """Should handle data shorter than indicator windows."""
        data = make_price_frame(rows=10)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert not result.empty

    def test_feature_names_returns_list(self) -> None:
        engineer = FeatureEngineer()
        names = engineer.feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_transform_with_known_data(self) -> None:
        """Test with simple known data to verify calculations."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "Open": np.full(30, 100.0),
                "High": np.full(30, 105.0),
                "Low": np.full(30, 95.0),
                "Close": np.full(30, 100.0),
                "Volume": 100_000,
            },
            index=dates,
        )
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert (result["SMA_20"].dropna() == 100.0).all()
        assert (result["EMA_12"].dropna() == 100.0).all()

    def test_bollinger_band_width_positive(self) -> None:
        """BB width should be positive when there is variation."""
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        valid = result["BB_width"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert (result["Drawdown"] <= 0).all()

    def test_cumulative_returns_starts_at_zero(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        assert result["Cumulative_returns"].iloc[0] == 0.0

    def test_atr_percent_positive(self) -> None:
        data = make_price_frame(rows=80)
        engineer = FeatureEngineer()
        result = engineer.transform(data)
        valid = result["ATR_percent"].dropna()
        assert (valid >= 0).all()
