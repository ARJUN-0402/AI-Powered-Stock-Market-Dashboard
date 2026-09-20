"""Tests for the quantitative analytics engine.

Tests cover all indicators with:
- Shape and alignment invariants
- Range checks where applicable
- NaN handling and propagation
- Edge cases (constant prices, insufficient history, single values)
- Mathematically expected values for known fixtures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.analytics import (
    adx,
    atr,
    bollinger_bands,
    cumulative_returns,
    drawdown,
    ema,
    fifty_two_week_high,
    fifty_two_week_low,
    macd,
    obv,
    rolling_volatility,
    rsi,
    sma,
    stochastic_oscillator,
    stochastic_rsi,
    vwap,
)
from tests.fixtures import make_constant_frame, make_price_frame

# =============================================================================
# SMA Tests
# =============================================================================


class TestSMA:
    def test_sma_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = sma(data, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_sma_first_values_are_nan(self) -> None:
        data = make_price_frame(rows=80)
        result = sma(data, window=20)
        assert result.iloc[:19].isna().all()

    def test_sma_manual_calculation(self) -> None:
        """Verify SMA against manual calculation."""
        data = make_price_frame(rows=30)
        result = sma(data, window=5)
        expected = data["Close"].rolling(window=5, min_periods=5).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_sma_constant_price(self) -> None:
        """SMA of constant price should equal that price."""
        data = make_constant_frame(rows=30, price=100.0)
        result = sma(data, window=10)
        valid = result.dropna()
        assert (valid == 100.0).all()

    def test_sma_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            sma(data)

    def test_sma_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            sma(data, window=0)

    def test_sma_window_one_equals_price(self) -> None:
        data = make_price_frame(rows=10)
        result = sma(data, window=1)
        pd.testing.assert_series_equal(result, data["Close"].astype(float), check_names=False)


# =============================================================================
# EMA Tests
# =============================================================================


class TestEMA:
    def test_ema_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = ema(data, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_ema_first_value_is_seed(self) -> None:
        """EMA should start from the first price value."""
        data = make_price_frame(rows=30)
        result = ema(data, window=10)
        assert result.iloc[0].item() == result.iloc[0].item()  # not NaN

    def test_ema_constant_price(self) -> None:
        """EMA of constant price should equal that price."""
        data = make_constant_frame(rows=30, price=100.0)
        result = ema(data, window=10)
        assert (result == 100.0).all()

    def test_ema_responds_to_trend(self) -> None:
        """EMA should increase when price increases."""
        data = make_price_frame(rows=50)
        result = ema(data, window=10)
        assert result.iloc[-1] > result.iloc[0]

    def test_ema_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            ema(data)

    def test_ema_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            ema(data, window=0)


# =============================================================================
# RSI Tests
# =============================================================================


class TestRSI:
    def test_rsi_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = rsi(data, window=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_rsi_values_in_range(self) -> None:
        data = make_price_frame(rows=80)
        result = rsi(data, window=14)
        valid = result.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_rsi_first_values_are_nan(self) -> None:
        data = make_price_frame(rows=80)
        result = rsi(data, window=14)
        assert result.iloc[:14].isna().all()

    def test_rsi_constant_price(self) -> None:
        """RSI of constant price should be 100 (no losses)."""
        data = make_constant_frame(rows=30, price=100.0)
        result = rsi(data, window=14)
        valid = result.dropna()
        assert (valid == 100.0).all()

    def test_rsi_all_gains(self) -> None:
        """RSI should approach 100 when price only goes up."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "Open": np.arange(1, 31, dtype=float),
                "High": np.arange(1, 31, dtype=float) + 0.5,
                "Low": np.arange(1, 31, dtype=float) - 0.5,
                "Close": np.arange(1, 31, dtype=float),
                "Volume": 100_000,
            },
            index=dates,
        )
        result = rsi(data, window=14)
        assert result.iloc[-1] == 100.0

    def test_rsi_all_losses(self) -> None:
        """RSI should approach 0 when price only goes down."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "Open": np.arange(30, 0, -1, dtype=float),
                "High": np.arange(30, 0, -1, dtype=float) + 0.5,
                "Low": np.arange(30, 0, -1, dtype=float) - 0.5,
                "Close": np.arange(30, 0, -1, dtype=float),
                "Volume": 100_000,
            },
            index=dates,
        )
        result = rsi(data, window=14)
        assert result.iloc[-1] == 0.0

    def test_rsi_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            rsi(data)

    def test_rsi_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            rsi(data, window=0)


# =============================================================================
# MACD Tests
# =============================================================================


class TestMACD:
    def test_macd_returns_dataframe_with_three_columns(self) -> None:
        data = make_price_frame(rows=80)
        result = macd(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_macd_histogram_is_difference(self) -> None:
        data = make_price_frame(rows=80)
        result = macd(data)
        expected_histogram = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected_histogram, check_names=False)

    def test_macd_length_matches_input(self) -> None:
        data = make_price_frame(rows=80)
        result = macd(data)
        assert len(result) == len(data)

    def test_macd_raises_on_invalid_windows(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            macd(data, short_window=26, long_window=12)

    def test_macd_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            macd(data)

    def test_macd_constant_price(self) -> None:
        """MACD of constant price should be 0."""
        data = make_constant_frame(rows=50, price=100.0)
        result = macd(data)
        valid = result.dropna()
        assert (valid["macd"] == 0.0).all()


# =============================================================================
# Bollinger Bands Tests
# =============================================================================


class TestBollingerBands:
    def test_bollinger_returns_dataframe_with_three_columns(self) -> None:
        data = make_price_frame(rows=80)
        result = bollinger_bands(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["middle", "upper", "lower"]

    def test_bollinger_upper_above_middle(self) -> None:
        data = make_price_frame(rows=80)
        result = bollinger_bands(data)
        valid = result.dropna()
        assert (valid["upper"] >= valid["middle"]).all()

    def test_bollinger_lower_below_middle(self) -> None:
        data = make_price_frame(rows=80)
        result = bollinger_bands(data)
        valid = result.dropna()
        assert (valid["lower"] <= valid["middle"]).all()

    def test_bollinger_constant_price(self) -> None:
        """Bollinger Bands of constant price should all equal that price."""
        data = make_constant_frame(rows=50, price=100.0)
        result = bollinger_bands(data)
        valid = result.dropna()
        assert (valid["upper"] == 100.0).all()
        assert (valid["lower"] == 100.0).all()

    def test_bollinger_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            bollinger_bands(data, window=0)

    def test_bollinger_raises_on_invalid_std(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            bollinger_bands(data, num_std=0)

    def test_bollinger_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            bollinger_bands(data)


# =============================================================================
# ATR Tests
# =============================================================================


class TestATR:
    def test_atr_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = atr(data, window=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_atr_is_non_negative(self) -> None:
        data = make_price_frame(rows=80)
        result = atr(data, window=14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_constant_price(self) -> None:
        """ATR of constant price should be 0."""
        data = make_constant_frame(rows=50, price=100.0)
        result = atr(data, window=14)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_atr_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["High"])
        with pytest.raises(KeyError):
            atr(data)

    def test_atr_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            atr(data, window=0)


# =============================================================================
# Stochastic Oscillator Tests
# =============================================================================


class TestStochasticOscillator:
    def test_stochastic_returns_dataframe_with_two_columns(self) -> None:
        data = make_price_frame(rows=80)
        result = stochastic_oscillator(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["%K", "%D"]

    def test_stochastic_k_in_range(self) -> None:
        data = make_price_frame(rows=80)
        result = stochastic_oscillator(data)
        valid = result["%K"].dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_stochastic_d_in_range(self) -> None:
        data = make_price_frame(rows=80)
        result = stochastic_oscillator(data)
        valid = result["%D"].dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_stochastic_constant_price(self) -> None:
        """Stochastic of constant price should be 0."""
        data = make_constant_frame(rows=50, price=100.0)
        result = stochastic_oscillator(data)
        valid = result["%K"].dropna()
        assert (valid == 0.0).all()

    def test_stochastic_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["High"])
        with pytest.raises(KeyError):
            stochastic_oscillator(data)

    def test_stochastic_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            stochastic_oscillator(data, k_window=0)


# =============================================================================
# Stochastic RSI Tests
# =============================================================================


class TestStochasticRSI:
    def test_stochastic_rsi_returns_dataframe_with_two_columns(self) -> None:
        data = make_price_frame(rows=80)
        result = stochastic_rsi(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["%K", "%D"]

    def test_stochastic_rsi_k_in_range(self) -> None:
        data = make_price_frame(rows=80)
        result = stochastic_rsi(data)
        valid = result["%K"].dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_stochastic_rsi_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            stochastic_rsi(data)


# =============================================================================
# ADX Tests
# =============================================================================


class TestADX:
    def test_adx_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = adx(data, window=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_adx_is_non_negative(self) -> None:
        data = make_price_frame(rows=80)
        result = adx(data, window=14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_adx_constant_price(self) -> None:
        """ADX of constant price should be 0 (no directional movement)."""
        data = make_constant_frame(rows=50, price=100.0)
        result = adx(data, window=14)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_adx_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["High"])
        with pytest.raises(KeyError):
            adx(data)

    def test_adx_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            adx(data, window=0)


# =============================================================================
# OBV Tests
# =============================================================================


class TestOBV:
    def test_obv_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = obv(data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_obv_first_value_is_zero(self) -> None:
        data = make_price_frame(rows=80)
        result = obv(data)
        assert result.iloc[0] == 0.0

    def test_obv_constant_price(self) -> None:
        """OBV of constant price should be 0 (no direction changes)."""
        data = make_constant_frame(rows=50, price=100.0)
        result = obv(data)
        assert (result == 0.0).all()

    def test_obv_increases_on_up_days(self) -> None:
        """OBV should increase when price goes up."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {
                "Open": np.arange(1, 11, dtype=float),
                "High": np.arange(1, 11, dtype=float) + 0.5,
                "Low": np.arange(1, 11, dtype=float) - 0.5,
                "Close": np.arange(1, 11, dtype=float),
                "Volume": 100_000,
            },
            index=dates,
        )
        result = obv(data)
        assert result.iloc[-1] > 0

    def test_obv_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Volume"])
        with pytest.raises(KeyError):
            obv(data)


# =============================================================================
# VWAP Tests
# =============================================================================


class TestVWAP:
    def test_vwap_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = vwap(data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_vwap_constant_price(self) -> None:
        """VWAP of constant price should equal that price."""
        data = make_constant_frame(rows=50, price=100.0)
        result = vwap(data)
        assert (result == 100.0).all()

    def test_vwap_within_overall_range(self) -> None:
        """Cumulative VWAP should be within the overall price range."""
        data = make_price_frame(rows=80)
        result = vwap(data)
        valid = result.dropna()
        assert (valid >= data["Close"].min()).all()
        assert (valid <= data["Close"].max()).all()

    def test_vwap_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Volume"])
        with pytest.raises(KeyError):
            vwap(data)


# =============================================================================
# Rolling Volatility Tests
# =============================================================================


class TestRollingVolatility:
    def test_volatility_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = rolling_volatility(data, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_volatility_is_non_negative(self) -> None:
        data = make_price_frame(rows=80)
        result = rolling_volatility(data, window=20)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_volatility_constant_price(self) -> None:
        """Volatility of constant price should be 0."""
        data = make_constant_frame(rows=50, price=100.0)
        result = rolling_volatility(data, window=20)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_volatility_annualized_is_larger(self) -> None:
        """Annualized volatility should be larger than raw."""
        data = make_price_frame(rows=80)
        raw = rolling_volatility(data, window=20, annualize=False)
        annualized = rolling_volatility(data, window=20, annualize=True)
        valid_idx = raw.dropna().index
        assert (annualized.loc[valid_idx] > raw.loc[valid_idx]).all()

    def test_volatility_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            rolling_volatility(data, window=1)

    def test_volatility_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            rolling_volatility(data)


# =============================================================================
# Cumulative Returns Tests
# =============================================================================


class TestCumulativeReturns:
    def test_cumulative_returns_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = cumulative_returns(data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_cumulative_returns_first_value_is_zero(self) -> None:
        data = make_price_frame(rows=80)
        result = cumulative_returns(data)
        assert result.iloc[0] == 0.0

    def test_cumulative_returns_constant_price(self) -> None:
        """Cumulative returns of constant price should be 0."""
        data = make_constant_frame(rows=50, price=100.0)
        result = cumulative_returns(data)
        assert (result == 0.0).all()

    def test_cumulative_returns_known_values(self) -> None:
        """Test against manually computed values."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100, 102, 104, 106, 108],
                "High": [101, 103, 105, 107, 109],
                "Low": [99, 101, 103, 105, 107],
                "Close": [100, 102, 104, 106, 108],
                "Volume": 100_000,
            },
            index=dates,
        )
        result = cumulative_returns(data)
        expected = pd.Series([0.0, 0.02, 0.04, 0.06, 0.08], index=dates)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_log_returns_known_values(self) -> None:
        """Test log returns against manually computed values."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100, 102, 104, 106, 108],
                "High": [101, 103, 105, 107, 109],
                "Low": [99, 101, 103, 105, 107],
                "Close": [100, 102, 104, 106, 108],
                "Volume": 100_000,
            },
            index=dates,
        )
        result = cumulative_returns(data, log_returns=True)
        expected = pd.Series(
            [0.0, np.log(1.02), np.log(1.04), np.log(1.06), np.log(1.08)],
            index=dates,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_cumulative_returns_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            cumulative_returns(data)


# =============================================================================
# Drawdown Tests
# =============================================================================


class TestDrawdown:
    def test_drawdown_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = drawdown(data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_drawdown_is_non_positive(self) -> None:
        data = make_price_frame(rows=80)
        result = drawdown(data)
        assert (result <= 0).all()

    def test_drawdown_constant_price(self) -> None:
        """Drawdown of constant price should be 0."""
        data = make_constant_frame(rows=50, price=100.0)
        result = drawdown(data)
        assert (result == 0.0).all()

    def test_drawdown_known_values(self) -> None:
        """Test drawdown against manually computed values."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100, 110, 105, 95, 100],
                "High": [101, 111, 106, 96, 101],
                "Low": [99, 109, 104, 94, 99],
                "Close": [100, 110, 105, 95, 100],
                "Volume": 100_000,
            },
            index=dates,
        )
        result = drawdown(data)
        # Running max: [100, 110, 110, 110, 110]
        # Drawdown: [0, 0, (105-110)/110, (95-110)/110, (100-110)/110]
        expected = pd.Series(
            [0.0, 0.0, -5 / 110, -15 / 110, -10 / 110],
            index=dates,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_drawdown_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            drawdown(data)


# =============================================================================
# 52-Week High/Low Tests
# =============================================================================


class TestFiftyTwoWeekHighLow:
    def test_fifty_two_week_high_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = fifty_two_week_high(data, window=252)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_fifty_two_week_low_returns_series_with_correct_length(self) -> None:
        data = make_price_frame(rows=80)
        result = fifty_two_week_low(data, window=252)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_high_always_above_low(self) -> None:
        data = make_price_frame(rows=80)
        high = fifty_two_week_high(data, window=20)
        low = fifty_two_week_low(data, window=20)
        assert (high >= low).all()

    def test_high_constant_price(self) -> None:
        data = make_constant_frame(rows=50, price=100.0)
        result = fifty_two_week_high(data, window=20)
        assert (result == 100.0).all()

    def test_low_constant_price(self) -> None:
        data = make_constant_frame(rows=50, price=100.0)
        result = fifty_two_week_low(data, window=20)
        assert (result == 100.0).all()

    def test_high_with_partial_window(self) -> None:
        """High should work with partial windows (min_periods=1)."""
        data = make_price_frame(rows=10)
        result = fifty_two_week_high(data, window=252)
        assert not result.isna().any()

    def test_low_with_partial_window(self) -> None:
        """Low should work with partial windows (min_periods=1)."""
        data = make_price_frame(rows=10)
        result = fifty_two_week_low(data, window=252)
        assert not result.isna().any()

    def test_high_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            fifty_two_week_high(data, window=0)

    def test_low_raises_on_invalid_window(self) -> None:
        data = make_price_frame(rows=30)
        with pytest.raises(ValueError):
            fifty_two_week_low(data, window=0)

    def test_high_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            fifty_two_week_high(data)

    def test_low_raises_on_missing_column(self) -> None:
        data = make_price_frame(rows=30).drop(columns=["Close"])
        with pytest.raises(KeyError):
            fifty_two_week_low(data)
