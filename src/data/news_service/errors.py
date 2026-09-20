"""Typed exception hierarchy for the news data layer.

Mirrors the :mod:`src.data.market_data_service.errors` hierarchy so the UI
can react to specific failure modes (rate limiting vs. authentication vs.
transport timeout) without depending on provider-specific exceptions.

A news provider that cannot authenticate, is rate limited, or times out is
never surfaced as an opaque stack trace: the service translates it into one of
the typed subclasses below so the dashboard can degrade gracefully and tell
the user *why* news is unavailable.
"""

from __future__ import annotations


class NewsDataError(RuntimeError):
    """Base class for every error raised by the news data layer."""


class InvalidSymbolError(NewsDataError, ValueError):
    """The supplied symbol is not a valid ticker."""


class NewsProviderError(NewsDataError):
    """The provider raised an unexpected error."""


class NewsAuthenticationError(NewsProviderError):
    """The provider requires credentials that are missing or invalid."""


class RateLimitedError(NewsProviderError):
    """The provider explicitly signalled that the caller is being throttled."""


class TimeoutExceededError(NewsProviderError):
    """The provider call exceeded the configured timeout."""
