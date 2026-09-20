"""Alert engine for technical indicators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.config import CONFIG
from src.features.technical_indicators import (
    calculate_moving_averages,
    calculate_rsi,
    latest_value,
)
from src.utils.logging import get_logger
from src.utils.numeric import is_nan
from src.utils.validation import safe_float

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for alerts surfaced to the UI."""

    INFO = "info"
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class Alert:
    """Single technical alert."""

    code: str
    message: str
    severity: AlertSeverity


def _rsi_alerts(data: pd.DataFrame) -> list[Alert]:
    if len(data) < CONFIG.rsi_period:
        return []
    rsi = calculate_rsi(data)
    current = safe_float(latest_value(rsi), default=float("nan"))
    if is_nan(current):
        return []
    if current < CONFIG.rsi_oversold:
        return [
            Alert(
                code="RSI_OVERSOLD",
                message=f" oversold (RSI: {current:.2f})",
                severity=AlertSeverity.POSITIVE,
            )
        ]
    if current > CONFIG.rsi_overbought:
        return [
            Alert(
                code="RSI_OVERBOUGHT",
                message=f" overbought (RSI: {current:.2f})",
                severity=AlertSeverity.NEGATIVE,
            )
        ]
    return []


def _moving_average_alerts(data: pd.DataFrame) -> list[Alert]:
    # MA_5 crossover detection only needs 6 rows of data.
    if len(data) < 6:
        return []
    mas = calculate_moving_averages(data, windows=(5,))
    if "MA_5" not in mas or len(mas["MA_5"]) < 2:
        return []

    current_price = safe_float(data["Close"].iloc[-1])
    previous_price = safe_float(data["Close"].iloc[-2])
    ma_current = safe_float(mas["MA_5"].iloc[-1])
    ma_previous = safe_float(mas["MA_5"].iloc[-2])

    alerts: list[Alert] = []
    if previous_price < ma_previous and current_price > ma_current:
        alerts.append(
            Alert(
                code="PRICE_CROSS_UP_MA5",
                message=f" price crossed above 5-day MA (${ma_current:.2f})",
                severity=AlertSeverity.POSITIVE,
            )
        )
    elif previous_price > ma_previous and current_price < ma_current:
        alerts.append(
            Alert(
                code="PRICE_CROSS_DOWN_MA5",
                message=f" price crossed below 5-day MA (${ma_current:.2f})",
                severity=AlertSeverity.NEGATIVE,
            )
        )
    return alerts


def generate_alerts(data: pd.DataFrame, symbol: str) -> list[Alert]:
    """Return all alerts triggered by ``data`` for ``symbol``.

    Parameters
    ----------
    data:
        OHLCV price frame.
    symbol:
        The ticker used in human-friendly messages.

    Returns
    -------
    list[Alert]
        A list of :class:`Alert` instances (possibly empty).
    """

    if data is None or data.empty:
        return []

    try:
        alerts = _rsi_alerts(data) + _moving_average_alerts(data)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to compute alerts for %s: %s", symbol, exc)
        return []
    return alerts


def alerts_have_positive(alerts: list[Alert]) -> bool:
    """Return ``True`` if any alert in ``alerts`` is bullish."""

    return any(alert.severity == AlertSeverity.POSITIVE for alert in alerts)


def alerts_have_negative(alerts: list[Alert]) -> bool:
    """Return ``True`` if any alert in ``alerts`` is bearish."""

    return any(alert.severity == AlertSeverity.NEGATIVE for alert in alerts)
