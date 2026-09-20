"""Technical indicator calculations.

All functions operate on a :class:`pandas.DataFrame` with at least the
``Close`` column. They return plain :class:`pandas.Series` or
:class:`pandas.DataFrame` objects without performing any I/O.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import CONFIG


def calculate_rsi(
    data: pd.DataFrame,
    window: int = CONFIG.rsi_period,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the Relative Strength Index.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window in periods.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        RSI values aligned to ``data.index``.
    """

    if price_column not in data.columns:
        raise KeyError(f"Column '{price_column}' not present in data")

    delta = data[price_column].diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    data: pd.DataFrame,
    short_window: int = CONFIG.macd_short,
    long_window: int = CONFIG.macd_long,
    signal_window: int = CONFIG.macd_signal,
    price_column: str = "Close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, the signal line and the histogram."""

    if price_column not in data.columns:
        raise KeyError(f"Column '{price_column}' not present in data")

    short_ema = data[price_column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[price_column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_moving_averages(
    data: pd.DataFrame,
    windows: Iterable[int] = CONFIG.moving_average_windows,
    price_column: str = "Close",
) -> dict[str, pd.Series]:
    """Return simple moving averages keyed by ``"MA_<window>"``."""

    if price_column not in data.columns:
        raise KeyError(f"Column '{price_column}' not present in data")

    mas: dict[str, pd.Series] = {}
    for window in windows:
        mas[f"MA_{window}"] = data[price_column].rolling(window=window, min_periods=window).mean()
    return mas


def latest_value(series: pd.Series) -> float:
    """Return the last element of ``series`` as ``float`` or ``float('nan')``."""

    if series is None or len(series) == 0:
        return float("nan")
    try:
        value = float(series.iloc[-1])
    except (TypeError, ValueError):
        return float("nan")
    from src.utils.numeric import is_nan

    return float("nan") if is_nan(value) else value
