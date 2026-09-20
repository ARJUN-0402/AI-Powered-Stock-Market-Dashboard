"""Data Transfer Objects returned by the market data service.

These objects are deliberately simple value types so they can be
serialised, cached and surfaced to the UI without leaking
provider-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class Quote:
    """A point-in-time quote for a single ticker.

    Attributes
    ----------
    symbol:
        The validated ticker symbol.
    price:
        The most recent price available. ``None`` if unknown.
    previous_close:
        The previous session's close. ``None`` if unknown.
    change:
        Absolute change versus ``previous_close``. ``0.0`` if a previous
        close is not available.
    change_pct:
        Percentage change versus ``previous_close``. ``0.0`` if a
        previous close is not available.
    volume:
        Most recent traded volume. ``0`` if unavailable.
    timestamp:
        Timestamp of the underlying observation.
    is_delayed:
        ``True`` when the price is known to be delayed (e.g. the
        provider explicitly indicates a delay). The UI must surface
        this so users do not mistake it for a real-time quote.
    is_available:
        ``True`` when the service has a usable observation. ``False``
        means callers should treat the quote as missing.
    currency:
        ISO currency code if known.
    provider:
        Identifier of the provider that produced the observation.
    """

    symbol: str
    price: float | None = None
    previous_close: float | None = None
    change: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    timestamp: datetime = field(default_factory=_utcnow)
    is_delayed: bool = True
    is_available: bool = False
    currency: str | None = None
    provider: str = ""

    @classmethod
    def unavailable(cls, symbol: str, provider: str = "") -> Quote:
        """Return a quote representing "no data"."""

        return cls(
            symbol=symbol,
            is_available=False,
            is_delayed=True,
            provider=provider,
        )


@dataclass(frozen=True)
class MarketStats:
    """High level statistics for a ticker.

    All monetary fields are ``None`` when the underlying data is not
    available. The dataclass never fabricates values.
    """

    symbol: str
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    previous_close: float | None = None
    volume: int = 0
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    market_cap: float | None = None
    currency: str | None = None
    is_delayed: bool = True
    is_available: bool = False
    provider: str = ""
    timestamp: datetime = field(default_factory=_utcnow)

    @classmethod
    def unavailable(cls, symbol: str, provider: str = "") -> MarketStats:
        """Return stats representing "no data"."""

        return cls(symbol=symbol, is_available=False, provider=provider)
