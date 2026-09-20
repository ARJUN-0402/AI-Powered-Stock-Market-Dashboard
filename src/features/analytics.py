"""Quantitative analytics engine for technical analysis.

This module provides robust, deterministic implementations of common technical
indicators. All functions operate on :class:`pandas.DataFrame` objects containing
OHLCV columns and return :class:`pandas.Series` or :class:`pandas.DataFrame`
objects aligned to the input index.

Design principles:
- **No look-ahead leakage**: each value depends only on data available at or
  before that timestamp.
- **Graceful degradation**: insufficient history produces ``NaN`` rather than
  raised exceptions.
- **NaN propagation**: ``NaN`` inputs produce ``NaN`` outputs.
- **Pure functions**: no I/O, no global state, no side effects.

Indicator reference
-------------------
- :func:`sma` - Simple Moving Average
- :func:`ema` - Exponential Moving Average
- :func:`rsi` - Relative Strength Index (Wilder)
- :func:`macd` - Moving Average Convergence Divergence
- :func:`bollinger_bands` - Bollinger Bands
- :func:`atr` - Average True Range
- :func:`stochastic_oscillator` - Stochastic Oscillator (%K, %D)
- :func:`stochastic_rsi` - Stochastic RSI
- :func:`adx` - Average Directional Index
- :func:`obv` - On-Balance Volume
- :func:`vwap` - Volume-Weighted Average Price
- :func:`rolling_volatility` - Rolling standard deviation of returns
- :func:`cumulative_returns` - Cumulative simple returns
- :func:`drawdown` - Drawdown from peak
- :func:`fifty_two_week_high` - 52-week rolling high
- :func:`fifty_two_week_low` - 52-week rolling low
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def _require_column(data: pd.DataFrame, column: str) -> None:
    """Raise ``KeyError`` if ``column`` is missing from ``data``."""
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not present in data")


def _safe_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    """Wrap values in a Series aligned to ``index``."""
    return pd.Series(values, index=index, dtype=float)


def sma(
    data: pd.DataFrame,
    window: int = 20,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the Simple Moving Average.

    The SMA is the arithmetic mean of the last ``window`` observations:

    .. math:: SMA_t = \\frac{1}{n}\\sum_{i=0}^{n-1} P_{t-i}

    where :math:`P` is the price series and :math:`n` is the window length.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window in periods. Must be positive.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        SMA values aligned to ``data.index``. The first ``window - 1`` entries
        are ``NaN``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window`` is less than 1.

    Notes
    -----
    - Produces ``NaN`` for any window containing ``NaN`` values.
    - Uses ``min_periods=window`` so partial windows are not averaged.
    """
    _require_column(data, price_column)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return data[price_column].rolling(window=window, min_periods=window).mean()


def ema(
    data: pd.DataFrame,
    window: int = 20,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the Exponential Moving Average.

    The EMA applies exponentially decaying weights to past observations:

    .. math::
        \\alpha &= \\frac{2}{n+1} \\\\
        EMA_t &= \\alpha \\cdot P_t + (1 - \\alpha) \\cdot EMA_{t-1}

    where :math:`n` is the window length. The first value seeds from the first
    available price observation (standard pandas ``ewm`` behaviour with
    ``adjust=False``).

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Span of the EMA in periods. Must be positive.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        EMA values aligned to ``data.index``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window`` is less than 1.

    Notes
    -----
    - Uses the standard ``adjust=False`` (recursive) formulation.
    - The first ``window - 1`` values are influenced by the seed and should be
      interpreted with caution.
    """
    _require_column(data, price_column)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return data[price_column].ewm(span=window, adjust=False).mean()


def rsi(
    data: pd.DataFrame,
    window: int = CONFIG.rsi_period,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the Relative Strength Index (Wilder's smoothing).

    RSI measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions:

    .. math::
        RS_t &= \\frac{\\text{Avg Gain}}{\\text{Avg Loss}} \\\\
        RSI_t &= 100 - \\frac{100}{1 + RS_t}

    Wilder's smoothing uses a rolling mean seeded from the first ``window``
    deltas and then recursively smoothed with :math:`\\alpha = 1/n`.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window in periods. Must be positive.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        RSI values in ``[0, 100]`` aligned to ``data.index``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window`` is less than 1.

    Notes
    -----
    - When all losses in the window are zero, RSI returns 100.
    - When all gains in the window are zero, RSI returns 0.
    - The first ``window`` entries are ``NaN`` (seed period).
    - ``NaN`` in the price column propagates through the calculation.
    """
    _require_column(data, price_column)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    delta = data[price_column].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values


def macd(
    data: pd.DataFrame,
    short_window: int = CONFIG.macd_short,
    long_window: int = CONFIG.macd_long,
    signal_window: int = CONFIG.macd_signal,
    price_column: str = "Close",
) -> pd.DataFrame:
    """Compute MACD, signal line, and histogram.

    .. math::
        MACD_t &= EMA_{short}(P_t) - EMA_{long}(P_t) \\\\
        Signal_t &= EMA_{signal}(MACD_t) \\\\
        Histogram_t &= MACD_t - Signal_t

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    short_window:
        Span of the fast EMA.
    long_window:
        Span of the slow EMA.
    signal_window:
        Span of the signal line EMA.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.DataFrame
        Three columns: ``macd``, ``signal``, ``histogram``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``short_window >= long_window`` or any window < 1.
    """
    _require_column(data, price_column)
    if short_window < 1 or long_window < 1 or signal_window < 1:
        raise ValueError("All windows must be >= 1")
    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be < long_window ({long_window})"
        )

    short_ema = data[price_column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[price_column].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }
    )


def bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_column: str = "Close",
) -> pd.DataFrame:
    """Compute Bollinger Bands.

    .. math::
        Middle_t &= SMA_{window}(P_t) \\\\
        Upper_t &= Middle_t + k \\cdot \\sigma_t \\\\
        Lower_t &= Middle_t - k \\cdot \\sigma_t

    where :math:`\\sigma_t` is the rolling standard deviation and :math:`k` is
    ``num_std``.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window for the moving average.
    num_std:
        Number of standard deviations for the bands.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.DataFrame
        Three columns: ``middle``, ``upper``, ``lower``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window < 1`` or ``num_std <= 0``.

    Notes
    -----
    - Uses the sample standard deviation (``ddof=1``) for unbiased estimation.
    - The first ``window - 1`` rows are ``NaN``.
    """
    _require_column(data, price_column)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if num_std <= 0:
        raise ValueError(f"num_std must be > 0, got {num_std}")

    middle = sma(data, window, price_column)
    rolling_std = data[price_column].rolling(window=window, min_periods=window).std(ddof=1)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
        }
    )


def atr(
    data: pd.DataFrame,
    window: int = 14,
) -> pd.Series:
    """Compute the Average True Range.

    True Range captures intra-period and gap volatility:

    .. math::
        TR_t = \\max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)

    The ATR is Wilder's smoothed average of TR:

    .. math::
        ATR_t = \\frac{ATR_{t-1} \\cdot (n-1) + TR_t}{n}

    Parameters
    ----------
    data:
        Price frame containing ``High``, ``Low``, ``Close``.
    window:
        Lookback window. Must be positive.

    Returns
    -------
    pandas.Series
        ATR values aligned to ``data.index``.

    Raises
    ------
    KeyError
        If ``High``, ``Low``, or ``Close`` is missing.
    ValueError
        If ``window < 1``.

    Notes
    -----
    - First value is the mean of the first ``window`` TR observations.
    - Subsequent values use Wilder's recursive smoothing.
    """
    for col in ("High", "Low", "Close"):
        _require_column(data, col)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def stochastic_oscillator(
    data: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """Compute the Stochastic Oscillator (%K and %D).

    .. math::
        \\%K_t = 100 \\cdot
            \\frac{C_t - L_{lowest}}{H_{highest} - L_{lowest}}

    where :math:`H_{highest}` and :math:`L_{lowest}` are the highest high and
    lowest low over the lookback window. :math:`\\%D` is the SMA of
    :math:`\\%K`.

    Parameters
    ----------
    data:
        Price frame containing ``High``, ``Low``, ``Close``.
    k_window:
        Lookback window for %K calculation.
    d_window:
        Smoothing window for %D (SMA of %K).

    Returns
    -------
    pd.DataFrame
        Two columns: ``%K``, ``%D``.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If ``k_window < 1`` or ``d_window < 1``.

    Notes
    -----
    - When highest high equals lowest low, %K is 0 (avoids division by zero).
    - The first ``k_window - 1`` rows are ``NaN``.
    """
    for col in ("High", "Low", "Close"):
        _require_column(data, col)
    if k_window < 1:
        raise ValueError(f"k_window must be >= 1, got {k_window}")
    if d_window < 1:
        raise ValueError(f"d_window must be >= 1, got {d_window}")

    lowest_low = data["Low"].rolling(window=k_window, min_periods=k_window).min()
    highest_high = data["High"].rolling(window=k_window, min_periods=k_window).max()

    range_hl = highest_high - lowest_low
    pct_k = 100 * (data["Close"] - lowest_low) / range_hl.replace(0, np.nan)
    pct_k = pct_k.where(range_hl != 0, 0.0)
    pct_d = pct_k.rolling(window=d_window, min_periods=d_window).mean()

    return pd.DataFrame(
        {
            "%K": pct_k,
            "%D": pct_d,
        }
    )


def stochastic_rsi(
    data: pd.DataFrame,
    rsi_window: int = 14,
    stoch_window: int = 14,
    d_window: int = 3,
    price_column: str = "Close",
) -> pd.DataFrame:
    """Compute Stochastic RSI.

    Applies the stochastic oscillator formula to RSI values instead of price:

    .. math::
        StochRSI_t = \\frac{RSI_t - RSI_{min}}{RSI_{max} - RSI_{min}}

    Output is scaled to ``[0, 100]``. %D is the SMA of %K.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    rsi_window:
        Window for the underlying RSI calculation.
    stoch_window:
        Window for the stochastic normalization.
    d_window:
        Smoothing window for %D.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pd.DataFrame
        Two columns: ``%K``, ``%D``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If any window < 1.

    Notes
    -----
    - More sensitive than standard RSI; oscillates between 0 and 100.
    - Double-smoothing (RSI then stochastic) increases lag.
    """
    _require_column(data, price_column)
    if rsi_window < 1 or stoch_window < 1 or d_window < 1:
        raise ValueError("All windows must be >= 1")

    rsi_values = rsi(data, rsi_window, price_column)

    rsi_min = rsi_values.rolling(window=stoch_window, min_periods=stoch_window).min()
    rsi_max = rsi_values.rolling(window=stoch_window, min_periods=stoch_window).max()
    range_rsi = rsi_max - rsi_min

    pct_k = 100 * (rsi_values - rsi_min) / range_rsi.replace(0, np.nan)
    pct_k = pct_k.where(range_rsi != 0, 0.0)
    pct_d = pct_k.rolling(window=d_window, min_periods=d_window).mean()

    return pd.DataFrame(
        {
            "%K": pct_k,
            "%D": pct_d,
        }
    )


def adx(
    data: pd.DataFrame,
    window: int = 14,
) -> pd.Series:
    """Compute the Average Directional Index.

    ADX quantifies trend strength (not direction):

    .. math::
        +DM_t &= H_t - H_{t-1} \\text{ (if positive and > -DM)} \\\\
        -DM_t &= L_{t-1} - L_t \\text{ (if positive and > +DM)} \\\\
        TR_t &= \\text{True Range} \\\\
        +DI &= 100 \\cdot EMA(+DM / TR) \\\\
        -DI &= 100 \\cdot EMA(-DM / TR) \\\\
        DX &= 100 \\cdot \\frac{|+DI - -DI|}{+DI + -DI} \\\\
        ADX &= EMA(DX)

    Parameters
    ----------
    data:
        Price frame containing ``High``, ``Low``, ``Close``.
    window:
        Lookback window. Must be positive.

    Returns
    -------
    pandas.Series
        ADX values in ``[0, 100]``.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If ``window < 1``.

    Notes
    -----
    - ADX > 25 often indicates a trending market (not a signal to trade).
    - ADX < 20 often indicates a range-bound market.
    - Requires ``2 * window`` rows for full stabilization.
    """
    for col in ("High", "Low", "Close"):
        _require_column(data, col)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=data.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=data.index,
    )

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_vals = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di_smooth = plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_di_smooth / atr_vals.replace(0, np.nan)
    minus_di = 100 * minus_di_smooth / atr_vals.replace(0, np.nan)

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)

    adx_values = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx_values


def obv(
    data: pd.DataFrame,
) -> pd.Series:
    """Compute On-Balance Volume.

    OBV accumulates volume based on price direction:

    .. math::
        OBV_t = OBV_{t-1} +
            \\begin{cases}
                +V_t & \\text{if } C_t > C_{t-1} \\\\
                -V_t & \\text{if } C_t < C_{t-1} \\\\
                0 & \\text{otherwise}
            \\end{cases}

    Parameters
    ----------
    data:
        Price frame containing ``Close`` and ``Volume``.

    Returns
    -------
    pandas.Series
        Cumulative OBV values. First value is 0.

    Raises
    ------
    KeyError
        If ``Close`` or ``Volume`` is missing.

    Notes
    -----
    - Absolute OBV values are meaningless; focus on the trend/slope.
    - Divergences between OBV and price may indicate weakening trends.
    """
    for col in ("Close", "Volume"):
        _require_column(data, col)

    direction = np.sign(data["Close"].diff())
    direction = direction.fillna(0)
    return (direction * data["Volume"]).cumsum()


def vwap(
    data: pd.DataFrame,
) -> pd.Series:
    """Compute the Volume-Weighted Average Price.

    VWAP is the cumulative average of price weighted by volume:

    .. math::
        VWAP_t = \\frac{\\sum (P_{typical} \\cdot V)}{\\sum V}

    where :math:`P_{typical} = (H + L + C) / 3`.

    Parameters
    ----------
    data:
        Price frame containing ``High``, ``Low``, ``Close``, ``Volume``.

    Returns
    -------
    pandas.Series
        Cumulative VWAP from the start of the series.

    Raises
    ------
    KeyError
        If required columns are missing.

    Notes
    -----
    - Resets meaningfully only when applied to intraday data within a session.
    - On daily data, it converges to a long-term average.
    """
    for col in ("High", "Low", "Close", "Volume"):
        _require_column(data, col)

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    return (typical_price * data["Volume"]).cumsum() / data["Volume"].cumsum().replace(0, np.nan)


def rolling_volatility(
    data: pd.DataFrame,
    window: int = 20,
    price_column: str = "Close",
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling standard deviation of log returns.

    .. math::
        \\sigma_t = \\text{std}(\\ln(P_{t-i} / P_{t-i-1})_{i=0}^{n-1})

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window in periods.
    price_column:
        Column to use for the calculation.
    annualize:
        If ``True``, multiply by :math:`\\sqrt{periods\\_per\\_year}`.
    periods_per_year:
        Number of trading periods per year (default 252 for daily).

    Returns
    -------
    pandas.Series
        Rolling volatility aligned to ``data.index``.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window < 2``.

    Notes
    -----
    - Uses log returns (continuously compounded) for better statistical
      properties.
    - ``min_periods=window`` ensures sufficient observations.
    """
    _require_column(data, price_column)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    log_returns = np.log(data[price_column] / data[price_column].shift(1))
    vol = log_returns.rolling(window=window, min_periods=window).std(ddof=1)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def cumulative_returns(
    data: pd.DataFrame,
    price_column: str = "Close",
    log_returns: bool = False,
) -> pd.Series:
    """Compute cumulative returns from the start of the series.

    Simple cumulative return:

    .. math:: CR_t = \\frac{P_t}{P_0} - 1

    Log cumulative return:

    .. math:: CR_t = \\ln\\left(\\frac{P_t}{P_0}\\right)

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    price_column:
        Column to use for the calculation.
    log_returns:
        If ``True``, compute cumulative log returns.

    Returns
    -------
    pandas.Series
        Cumulative returns. First value is 0.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.

    Notes
    -----
    - Simple returns are additive across assets; log returns are additive
      across time.
    """
    _require_column(data, price_column)

    if log_returns:
        return np.log(data[price_column] / data[price_column].iloc[0])
    return (data[price_column] / data[price_column].iloc[0]) - 1


def drawdown(
    data: pd.DataFrame,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the drawdown from the running peak.

    .. math:: DD_t = \\frac{P_t}{\\max_{i \\leq t} P_i} - 1

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        Drawdown values in ``(-1, 0]``. 0 means at a new high.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.

    Notes
    -----
    - Always non-positive; 0 indicates the price is at its running maximum.
    - Maximum drawdown is the minimum value of this series.
    """
    _require_column(data, price_column)

    running_max = data[price_column].cummax()
    return (data[price_column] / running_max) - 1


def fifty_two_week_high(
    data: pd.DataFrame,
    window: int = 252,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the rolling 52-week high.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window. Default 252 (approximate trading days per year).
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        Rolling maximum over ``window`` periods.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window < 1``.

    Notes
    -----
    - Assumes daily data. For weekly data, use ``window=52``.
    - ``min_periods=1`` ensures values are produced even with limited history.
    """
    _require_column(data, price_column)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return data[price_column].rolling(window=window, min_periods=1).max()


def fifty_two_week_low(
    data: pd.DataFrame,
    window: int = 252,
    price_column: str = "Close",
) -> pd.Series:
    """Compute the rolling 52-week low.

    Parameters
    ----------
    data:
        Price frame containing ``price_column``.
    window:
        Lookback window. Default 252 (approximate trading days per year).
    price_column:
        Column to use for the calculation.

    Returns
    -------
    pandas.Series
        Rolling minimum over ``window`` periods.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If ``window < 1``.

    Notes
    -----
    - Assumes daily data. For weekly data, use ``window=52``.
    - ``min_periods=1`` ensures values are produced even with limited history.
    """
    _require_column(data, price_column)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return data[price_column].rolling(window=window, min_periods=1).min()
