"""Tests for technical indicator helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features.technical_indicators import (
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    latest_value,
)
from tests.fixtures import make_price_frame


def test_calculate_rsi_returns_series_with_expected_length() -> None:
    data = make_price_frame(rows=80)
    rsi = calculate_rsi(data)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(data)
    # The first ``window``-1 entries must be NaN, the rest must be within [0, 100].
    assert rsi.iloc[:13].isna().all()
    valid = rsi.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_calculate_rsi_raises_on_missing_column() -> None:
    data = make_price_frame(rows=80).drop(columns=["Close"])
    with pytest.raises(KeyError):
        calculate_rsi(data)


def test_calculate_macd_returns_three_series() -> None:
    data = make_price_frame(rows=120)
    macd, signal, histogram = calculate_macd(data)
    assert len(macd) == len(signal) == len(histogram) == len(data)
    np.testing.assert_allclose(
        (macd - signal).dropna().to_numpy(),
        histogram.dropna().to_numpy(),
    )


def test_calculate_moving_averages_keys_match_windows() -> None:
    data = make_price_frame(rows=80)
    mas = calculate_moving_averages(data, windows=[5, 20])
    assert set(mas.keys()) == {"MA_5", "MA_20"}
    for series in mas.values():
        assert isinstance(series, pd.Series)


def test_calculate_moving_averages_raises_on_missing_column() -> None:
    data = make_price_frame(rows=80).drop(columns=["Close"])
    with pytest.raises(KeyError):
        calculate_moving_averages(data, windows=[5])


def test_latest_value_handles_empty_and_nan() -> None:
    assert math.isnan(latest_value(pd.Series(dtype=float)))
    series = pd.Series([1.0, 2.0, float("nan")])
    assert math.isnan(latest_value(series))
    assert latest_value(pd.Series([1.0, 2.0, 3.0])) == 3.0
