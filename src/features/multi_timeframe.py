"""Multi-timeframe analytics support.

This module provides utilities for computing technical indicators across
multiple timeframes (daily, weekly, intraday) and combining them into a
unified feature set.

Timeframe conventions
---------------------
- **Daily**: Standard daily bars (default for most analyses).
- **Weekly**: Aggregated weekly bars (Friday-to-Friday or last available).
- **Intraday**: Sub-daily bars (requires sufficient intraday history).

All resampling is done using standard OHLCV aggregation rules to avoid
look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.analytics import (
    fifty_two_week_high,
    fifty_two_week_low,
    rolling_volatility,
)
from src.features.feature_engineer import FeatureEngineer


def resample_to_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly frequency.

    Uses Friday as the week end. If Friday is not a trading day, the last
    available day of the week is used.

    Parameters
    ----------
    data:
        Daily OHLCV DataFrame with a DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        Weekly OHLCV DataFrame.

    Raises
    ------
    ValueError
        If ``data`` is empty or has no DatetimeIndex.

    Notes
    -----
    - Volume is summed across the week.
    - Open is the first open of the week; Close is the last close.
    """
    if data.empty:
        raise ValueError("Cannot resample empty DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex")

    return data.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()


def aggregate_intraday(data: pd.DataFrame, target: str = "5min") -> pd.DataFrame:
    """Aggregate intraday bars to a higher timeframe.

    Parameters
    ----------
    data:
        Intraday OHLCV DataFrame with a DatetimeIndex.
    target:
        Target frequency string (e.g., ``"5min"``, ``"15min"``, ``"1h"``).

    Returns
    -------
    pandas.DataFrame
        Aggregated OHLCV DataFrame.

    Raises
    ------
    ValueError
        If ``data`` is empty or has no DatetimeIndex.

    Notes
    -----
    - Useful for normalizing mixed-frequency data.
    - Volume is summed across the aggregation window.
    """
    if data.empty:
        raise ValueError("Cannot aggregate empty DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex")

    return data.resample(target).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()


@dataclass
class MultiTimeframeEngineer:
    """Compute features across multiple timeframes.

    Produces a combined feature set with timeframe-specific prefixes to
    distinguish indicators computed on different resolutions.

    Parameters
    ----------
    daily_engineer:
        FeatureEngineer for daily timeframe.
    weekly_engineer:
        FeatureEngineer for weekly timeframe.
    include_weekly:
        Whether to include weekly features (requires sufficient history).
    """

    daily_engineer: FeatureEngineer = None  # type: ignore[assignment]
    weekly_engineer: FeatureEngineer = None  # type: ignore[assignment]
    include_weekly: bool = True

    def __post_init__(self) -> None:
        if self.daily_engineer is None:
            self.daily_engineer = FeatureEngineer()
        if self.weekly_engineer is None:
            self.weekly_engineer = FeatureEngineer(
                high_low_window=52,
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute multi-timeframe features.

        Parameters
        ----------
        data:
            Daily OHLCV DataFrame.

        Returns
        -------
        pandas.DataFrame
            Combined features with prefixes ``daily_`` and ``weekly_``.

        Raises
        ------
        ValueError
            If ``data`` is empty.
        """
        if data.empty:
            raise ValueError("Cannot transform empty DataFrame")

        daily_features = self.daily_engineer.transform(data)
        daily_cols = [c for c in daily_features.columns if c not in data.columns]
        result = daily_features.rename(columns={c: f"daily_{c}" for c in daily_cols})

        if self.include_weekly and len(data) >= 10:
            try:
                weekly_data = resample_to_weekly(data)
                if len(weekly_data) >= 10:
                    weekly_features = self.weekly_engineer.transform(weekly_data)
                    weekly_cols = [
                        c for c in weekly_features.columns if c not in weekly_data.columns
                    ]
                    weekly_renamed = weekly_features.rename(
                        columns={c: f"weekly_{c}" for c in weekly_cols}
                    )
                    weekly_aligned = weekly_renamed.reindex(data.index, method="ffill")
                    result = result.join(weekly_aligned, how="left")
            except (ValueError, KeyError):
                pass

        return result


def compute_timeframe_summary(
    data: pd.DataFrame,
    window: int = 252,
) -> dict[str, float | None]:
    """Compute summary statistics for the current timeframe.

    Parameters
    ----------
    data:
        OHLCV DataFrame.
    window:
        Lookback window for high/low calculations.

    Returns
    -------
    dict
        Dictionary with keys: ``current_price``, ``daily_return``,
        ``weekly_return``, ``monthly_return``, ``volatility``,
        ``distance_from_52w_high``, ``distance_from_52w_low``.
    """
    if data.empty or len(data) < 2:
        return {
            "current_price": None,
            "daily_return": None,
            "weekly_return": None,
            "monthly_return": None,
            "volatility": None,
            "distance_from_52w_high": None,
            "distance_from_52w_low": None,
        }

    close = data["Close"]
    current = close.iloc[-1]

    daily_return = (close.iloc[-1] / close.iloc[-2]) - 1 if len(close) >= 2 else None
    weekly_return = (close.iloc[-1] / close.iloc[-5]) - 1 if len(close) >= 5 else None
    monthly_return = (close.iloc[-1] / close.iloc[-21]) - 1 if len(close) >= 21 else None

    vol_series = rolling_volatility(data, window=min(20, len(data) - 1))
    volatility = vol_series.iloc[-1] if not vol_series.empty else None

    high_52w = fifty_two_week_high(data, window=window).iloc[-1]
    low_52w = fifty_two_week_low(data, window=window).iloc[-1]

    return {
        "current_price": current,
        "daily_return": daily_return,
        "weekly_return": weekly_return,
        "monthly_return": monthly_return,
        "volatility": volatility,
        "distance_from_52w_high": (current / high_52w) - 1,
        "distance_from_52w_low": (current / low_52w) - 1,
    }
