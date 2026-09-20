"""Feature engineering API for quantitative analytics.

This module provides a unified interface for computing technical-analysis features
from OHLCV price data. The :class:`FeatureEngineer` class encapsulates all
indicator calculations and returns a single DataFrame with standardized column
names.

Example
-------
>>> from src.features.feature_engineer import FeatureEngineer
>>> features = FeatureEngineer().transform(price_data)
>>> print(features.columns)

The resulting DataFrame contains all computed features aligned to the input
index, making it suitable for downstream ML pipelines or visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import CONFIG
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


@dataclass
class FeatureEngineer:
    """Compute a comprehensive set of technical-analysis features.

    Parameters
    ----------
    rsi_window:
        Lookback window for RSI.
    macd_short:
        Fast EMA span for MACD.
    macd_long:
        Slow EMA span for MACD.
    macd_signal:
        Signal line EMA span for MACD.
    bollinger_window:
        Window for Bollinger Bands.
    bollinger_std:
        Number of standard deviations for Bollinger Bands.
    atr_window:
        Window for ATR.
    stochastic_k:
        Window for Stochastic %K.
    stochastic_d:
        Window for Stochastic %D.
    adx_window:
        Window for ADX.
    volatility_window:
        Window for rolling volatility.
    sma_windows:
        Windows for SMA features.
    ema_windows:
        Windows for EMA features.
    high_low_window:
        Window for 52-week high/low calculations.
    """

    rsi_window: int = CONFIG.rsi_period
    macd_short: int = CONFIG.macd_short
    macd_long: int = CONFIG.macd_long
    macd_signal: int = CONFIG.macd_signal
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_window: int = 14
    stochastic_k: int = 14
    stochastic_d: int = 3
    adx_window: int = 14
    volatility_window: int = 20
    sma_windows: tuple[int, ...] = (20, 50, 200)
    ema_windows: tuple[int, ...] = (12, 26)
    high_low_window: int = 252

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all features and return a single DataFrame.

        Parameters
        ----------
        data:
            OHLCV DataFrame with columns ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume``.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing all computed features. The original OHLCV
            columns are preserved.

        Raises
        ------
        KeyError
            If required columns are missing from ``data``.
        ValueError
            If ``data`` is empty.

        Notes
        -----
        - All features are aligned to ``data.index``.
        - NaN values appear where insufficient history exists.
        - No look-ahead leakage: each value depends only on past data.
        """
        if data.empty:
            raise ValueError("Cannot transform empty DataFrame")

        result = data.copy()

        result = self._add_moving_averages(result)
        result = self._add_rsi(result)
        result = self._add_macd(result)
        result = self._add_bollinger_bands(result)
        result = self._add_atr(result)
        result = self._add_stochastic(result)
        result = self._add_stochastic_rsi(result)
        result = self._add_adx(result)
        result = self._add_volume_features(result)
        result = self._add_volatility(result)
        result = self._add_returns(result)
        result = self._add_drawdown(result)
        result = self._add_high_low(result)

        return result

    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and EMA features."""
        for window in self.sma_windows:
            data[f"SMA_{window}"] = sma(data, window=window)
        for window in self.ema_windows:
            data[f"EMA_{window}"] = ema(data, window=window)
        return data

    def _add_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add RSI feature."""
        data["RSI"] = rsi(data, window=self.rsi_window)
        return data

    def _add_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD features."""
        macd_df = macd(
            data,
            short_window=self.macd_short,
            long_window=self.macd_long,
            signal_window=self.macd_signal,
        )
        data["MACD"] = macd_df["macd"]
        data["MACD_signal"] = macd_df["signal"]
        data["MACD_histogram"] = macd_df["histogram"]
        return data

    def _add_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Band features."""
        bb = bollinger_bands(
            data,
            window=self.bollinger_window,
            num_std=self.bollinger_std,
        )
        data["BB_middle"] = bb["middle"]
        data["BB_upper"] = bb["upper"]
        data["BB_lower"] = bb["lower"]
        data["BB_width"] = (bb["upper"] - bb["lower"]) / bb["middle"]
        data["BB_percent"] = (data["Close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
        return data

    def _add_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ATR feature."""
        data["ATR"] = atr(data, window=self.atr_window)
        data["ATR_percent"] = data["ATR"] / data["Close"] * 100
        return data

    def _add_stochastic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator features."""
        stoch = stochastic_oscillator(
            data,
            k_window=self.stochastic_k,
            d_window=self.stochastic_d,
        )
        data["Stoch_K"] = stoch["%K"]
        data["Stoch_D"] = stoch["%D"]
        return data

    def _add_stochastic_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic RSI features."""
        stoch_rsi = stochastic_rsi(
            data,
            rsi_window=self.rsi_window,
            stoch_window=self.stochastic_k,
            d_window=self.stochastic_d,
        )
        data["StochRSI_K"] = stoch_rsi["%K"]
        data["StochRSI_D"] = stoch_rsi["%D"]
        return data

    def _add_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ADX feature."""
        data["ADX"] = adx(data, window=self.adx_window)
        return data

    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        data["OBV"] = obv(data)
        data["VWAP"] = vwap(data)
        data["Volume_SMA_20"] = sma(data, window=20, price_column="Volume")
        data["Volume_ratio"] = data["Volume"] / data["Volume_SMA_20"]
        return data

    def _add_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling volatility features."""
        data["Volatility"] = rolling_volatility(data, window=self.volatility_window)
        data["Volatility_annualized"] = rolling_volatility(
            data, window=self.volatility_window, annualize=True
        )
        return data

    def _add_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add return features."""
        data["Returns_1d"] = data["Close"].pct_change()
        data["Returns_5d"] = data["Close"].pct_change(5)
        data["Returns_21d"] = data["Close"].pct_change(21)
        data["Cumulative_returns"] = cumulative_returns(data)
        data["Log_returns"] = cumulative_returns(data, log_returns=True)
        return data

    def _add_drawdown(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add drawdown feature."""
        data["Drawdown"] = drawdown(data)
        return data

    def _add_high_low(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add 52-week high/low features."""
        data["High_52w"] = fifty_two_week_high(data, window=self.high_low_window)
        data["Low_52w"] = fifty_two_week_low(data, window=self.high_low_window)
        data["Distance_from_high"] = (data["Close"] / data["High_52w"]) - 1
        data["Distance_from_low"] = (data["Close"] / data["Low_52w"]) - 1
        return data

    def feature_names(self) -> list[str]:
        """Return the list of feature column names produced by :meth:`transform`.

        This is useful for inspecting the output schema without running the
        full transformation.
        """
        sma_names = [f"SMA_{w}" for w in self.sma_windows]
        ema_names = [f"EMA_{w}" for w in self.ema_windows]
        return [
            *sma_names,
            *ema_names,
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_histogram",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "BB_width",
            "BB_percent",
            "ATR",
            "ATR_percent",
            "Stoch_K",
            "Stoch_D",
            "StochRSI_K",
            "StochRSI_D",
            "ADX",
            "OBV",
            "VWAP",
            "Volume_SMA_20",
            "Volume_ratio",
            "Volatility",
            "Volatility_annualized",
            "Returns_1d",
            "Returns_5d",
            "Returns_21d",
            "Cumulative_returns",
            "Log_returns",
            "Drawdown",
            "High_52w",
            "Low_52w",
            "Distance_from_high",
            "Distance_from_low",
        ]
