"""Tests for multi-timeframe analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.multi_timeframe import (
    MultiTimeframeEngineer,
    aggregate_intraday,
    compute_timeframe_summary,
    resample_to_weekly,
)
from tests.fixtures import make_price_frame


class TestResampleToWeekly:
    def test_resample_returns_dataframe(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        assert isinstance(result, pd.DataFrame)

    def test_resample_has_required_columns(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_resample_fewer_rows_than_daily(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        assert len(result) < len(data)

    def test_resample_open_is_first(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        first_week = data.iloc[:5]
        assert result.iloc[0]["Open"] == first_week.iloc[0]["Open"]

    def test_resample_close_is_last(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        first_week = data.iloc[:5]
        assert result.iloc[0]["Close"] == first_week.iloc[-1]["Close"]

    def test_resample_high_is_max(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        first_week = data.iloc[:5]
        assert result.iloc[0]["High"] == first_week["High"].max()

    def test_resample_low_is_min(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        first_week = data.iloc[:5]
        assert result.iloc[0]["Low"] == first_week["Low"].min()

    def test_resample_volume_is_sum(self) -> None:
        data = make_price_frame(rows=80)
        result = resample_to_weekly(data)
        first_week = data.iloc[:5]
        assert result.iloc[0]["Volume"] == first_week["Volume"].sum()

    def test_resample_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            resample_to_weekly(pd.DataFrame())

    def test_resample_raises_on_non_datetime_index(self) -> None:
        data = make_price_frame(rows=80)
        data.index = range(80)
        with pytest.raises(ValueError):
            resample_to_weekly(data)


class TestAggregateIntraday:
    def test_aggregate_returns_dataframe(self) -> None:
        dates = pd.date_range("2024-01-01 09:30", periods=100, freq="5min")
        data = pd.DataFrame(
            {
                "Open": np.random.default_rng(42).uniform(99, 101, 100),
                "High": np.random.default_rng(42).uniform(100, 102, 100),
                "Low": np.random.default_rng(42).uniform(98, 100, 100),
                "Close": np.random.default_rng(42).uniform(99, 101, 100),
                "Volume": np.random.default_rng(7).integers(1000, 5000, 100),
            },
            index=dates,
        )
        result = aggregate_intraday(data, target="15min")
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(data)

    def test_aggregate_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            aggregate_intraday(pd.DataFrame())


class TestMultiTimeframeEngineer:
    def test_transform_returns_dataframe(self) -> None:
        data = make_price_frame(rows=80)
        engineer = MultiTimeframeEngineer()
        result = engineer.transform(data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_has_daily_prefix(self) -> None:
        data = make_price_frame(rows=80)
        engineer = MultiTimeframeEngineer()
        result = engineer.transform(data)
        assert any(col.startswith("daily_") for col in result.columns)

    def test_transform_preserves_original_columns(self) -> None:
        data = make_price_frame(rows=80)
        engineer = MultiTimeframeEngineer()
        result = engineer.transform(data)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_transform_raises_on_empty(self) -> None:
        engineer = MultiTimeframeEngineer()
        with pytest.raises(ValueError):
            engineer.transform(pd.DataFrame())

    def test_transform_insufficient_data_no_weekly(self) -> None:
        """With very little data, weekly features should be skipped gracefully."""
        data = make_price_frame(rows=5)
        engineer = MultiTimeframeEngineer()
        result = engineer.transform(data)
        assert not result.empty


class TestComputeTimeframeSummary:
    def test_summary_returns_dict(self) -> None:
        data = make_price_frame(rows=80)
        result = compute_timeframe_summary(data)
        assert isinstance(result, dict)

    def test_summary_has_expected_keys(self) -> None:
        data = make_price_frame(rows=80)
        result = compute_timeframe_summary(data)
        expected_keys = {
            "current_price",
            "daily_return",
            "weekly_return",
            "monthly_return",
            "volatility",
            "distance_from_52w_high",
            "distance_from_52w_low",
        }
        assert set(result.keys()) == expected_keys

    def test_summary_empty_data(self) -> None:
        result = compute_timeframe_summary(pd.DataFrame())
        assert all(v is None for v in result.values())

    def test_summary_single_row(self) -> None:
        data = make_price_frame(rows=1)
        result = compute_timeframe_summary(data)
        assert all(v is None for v in result.values())

    def test_summary_current_price(self) -> None:
        data = make_price_frame(rows=80)
        result = compute_timeframe_summary(data)
        assert result["current_price"] == data["Close"].iloc[-1]
