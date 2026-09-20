"""Validation helpers for user supplied inputs."""

from __future__ import annotations

import math
from collections.abc import Iterable

from src.config import CONFIG


class ValidationError(ValueError):
    """Raised when a value fails validation."""


def validate_symbol(symbol: str) -> str:
    """Validate a ticker symbol.

    Parameters
    ----------
    symbol:
        Ticker symbol provided by a user.

    Returns
    -------
    str
        Normalised, uppercased ticker.

    Raises
    ------
    ValidationError
        If the symbol is empty or contains invalid characters.
    """

    if symbol is None:
        raise ValidationError("Stock symbol cannot be empty.")

    cleaned = str(symbol).strip().upper()
    if not cleaned:
        raise ValidationError("Stock symbol cannot be empty.")
    if not cleaned.replace(".", "").replace("-", "").isalnum():
        raise ValidationError(f"Stock symbol '{symbol}' contains invalid characters.")
    return cleaned


def validate_period(period: str) -> str:
    """Validate a yfinance period value."""

    if period not in CONFIG.valid_periods:
        raise ValidationError(f"Invalid period '{period}'. Allowed: {sorted(CONFIG.valid_periods)}")
    return period


def validate_interval(interval: str) -> str:
    """Validate a yfinance interval value."""

    if interval not in CONFIG.valid_intervals:
        raise ValidationError(
            f"Invalid interval '{interval}'. Allowed: {sorted(CONFIG.valid_intervals)}"
        )
    return interval


def validate_symbols(symbols: Iterable[str]) -> list[str]:
    """Validate and normalise an iterable of symbols."""

    return [validate_symbol(s) for s in symbols if s]


def safe_float(value, default: float = 0.0) -> float:
    """Convert ``value`` to ``float`` returning ``default`` on failure."""

    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default
