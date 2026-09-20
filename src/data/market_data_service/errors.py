"""Typed exception hierarchy for the market data layer.

The hierarchy is intentionally narrow so the UI can react to specific
failure modes (rate limiting vs. invalid symbol vs. transport timeout)
without depending on provider-specific exceptions.
"""

from __future__ import annotations


class MarketDataError(RuntimeError):
    """Base class for every error raised by the market data layer."""


class InvalidSymbolError(MarketDataError, ValueError):
    """The supplied symbol is not a valid ticker."""


class SymbolNotFoundError(MarketDataError):
    """The provider returned no data for an otherwise valid symbol."""


class ProviderError(MarketDataError):
    """The provider raised an unexpected error."""


class RateLimitedError(ProviderError):
    """The provider explicitly signalled that the caller is being throttled."""


class TimeoutExceededError(ProviderError):
    """The provider call exceeded the configured timeout."""
