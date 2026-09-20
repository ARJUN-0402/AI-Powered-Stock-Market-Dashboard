"""Centralised market data service.

:class:`MarketDataService` is the single entry point used by the rest
of the application to obtain historical prices, the latest quote and
high level statistics. It coordinates:

* Input validation (ticker symbols, periods, intervals).
* Provider invocation (delegated to a :class:`MarketDataProvider`).
* Result validation (OHLCV frame sanity checks).
* Caching (in-process TTL cache with negative caching for known
  invalid symbols).
* Error translation (provider exceptions are normalised into the
  service error hierarchy).

The service never fabricates data: a failure to retrieve a value is
surfaced to the caller either as an exception or as a DTO with
``is_available=False``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import CONFIG
from src.data.market_data_service.cache import TTLCache, register_negative_sentinel
from src.data.market_data_service.dto import MarketStats, Quote
from src.data.market_data_service.errors import (
    InvalidSymbolError,
    MarketDataError,
    SymbolNotFoundError,
)
from src.data.market_data_service.providers.base import MarketDataProvider
from src.data.market_data_service.providers.yfinance_provider import YFinanceProvider
from src.data.market_data_service.validators import (
    normalise_ohlcv_frame,
    require_valid_frame,
    safe_last_close,
    safe_previous_close,
)
from src.utils.logging import get_logger
from src.utils.validation import (
    ValidationError,
    validate_interval,
    validate_period,
    validate_symbol,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDataServiceConfig:
    """Runtime configuration for the service.

    All fields have sensible defaults sourced from
    :class:`src.config.AppConfig` but can be overridden individually
    for tests or specialised deployments.
    """

    history_ttl_seconds: float = field(default=float(CONFIG.cache_ttl_seconds))
    quote_ttl_seconds: float = field(default=float(CONFIG.live_price_ttl_seconds))
    stats_ttl_seconds: float = field(default=float(CONFIG.cache_ttl_seconds))
    negative_ttl_seconds: float = 30.0
    min_history_rows: int = 2


# Register sentinel types so the cache knows to use the short negative
# TTL for "no data" responses.
register_negative_sentinel(Quote)
register_negative_sentinel(MarketStats)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MarketDataService:
    """Reusable, provider-agnostic market data orchestrator."""

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        *,
        config: MarketDataServiceConfig | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        self._provider: MarketDataProvider = provider or YFinanceProvider()
        self._config = config or MarketDataServiceConfig()
        # One shared cache keeps concurrent streamlit users from
        # triggering the same upstream call twice.
        self._cache: TTLCache = cache or TTLCache(
            default_ttl_seconds=self._config.history_ttl_seconds,
            negative_ttl_seconds=self._config.negative_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def cache_stats(self) -> dict[str, int]:
        return self._cache.stats

    def configure(
        self,
        *,
        history_ttl_seconds: float | None = None,
        quote_ttl_seconds: float | None = None,
        stats_ttl_seconds: float | None = None,
        negative_ttl_seconds: float | None = None,
        min_history_rows: int | None = None,
    ) -> None:
        """Mutate service configuration in-place.

        ``history_ttl_seconds``, ``quote_ttl_seconds`` and
        ``negative_ttl_seconds`` only affect entries cached *after* the
        call. To clear the existing cache use :meth:`refresh`.
        """

        updates: dict[str, Any] = {}
        if history_ttl_seconds is not None:
            updates["history_ttl_seconds"] = float(history_ttl_seconds)
        if quote_ttl_seconds is not None:
            updates["quote_ttl_seconds"] = float(quote_ttl_seconds)
        if stats_ttl_seconds is not None:
            updates["stats_ttl_seconds"] = float(stats_ttl_seconds)
        if negative_ttl_seconds is not None:
            updates["negative_ttl_seconds"] = float(negative_ttl_seconds)
        if min_history_rows is not None:
            updates["min_history_rows"] = int(min_history_rows)
        if updates:
            self._config = MarketDataServiceConfig(**{**self._config.__dict__, **updates})

    def refresh(self, symbol: str | None = None) -> None:
        """Invalidate cached entries.

        Passing a specific ``symbol`` clears only entries for that
        symbol; without a value the whole cache is cleared.
        """

        if symbol is None:
            self._cache.clear()
            logger.debug("Market data cache cleared")
            return
        symbol_key = self._safe_symbol_key(symbol)
        if symbol_key is None:
            return
        for prefix in ("history", "quote", "stats"):
            self._cache.invalidate_prefix(f"{prefix}:{symbol_key}:")
        self._cache.invalidate(f"quote:{symbol_key}")
        self._cache.invalidate(f"stats:{symbol_key}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(
        self,
        symbol: str,
        *,
        period: str | None = None,
        interval: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return an OHLCV frame for ``symbol``.

        Raises
        ------
        InvalidSymbolError
            When the symbol is not a valid ticker.
        SymbolNotFoundError
            When the provider returns no usable data.
        MarketDataError
            For any other upstream failure.
        """

        normalised_symbol, normalised_period, normalised_interval = self._validate_request(
            symbol,
            period,
            interval,
        )
        cache_key = (
            f"history:{normalised_symbol}:{normalised_period}:{normalised_interval}"
        )

        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                logger.debug("Cache hit for %s", cache_key)
                return cached.copy()

        logger.info(
            "Fetching history for %s period=%s interval=%s (provider=%s)",
            normalised_symbol,
            normalised_period,
            normalised_interval,
            self._provider.name,
        )
        try:
            raw = self._provider.get_history(
                normalised_symbol,
                normalised_period,
                normalised_interval,
            )
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected provider error for %s: %s", normalised_symbol, exc)
            raise MarketDataError(f"Failed to fetch history for '{normalised_symbol}'") from exc

        frame = normalise_ohlcv_frame(raw)
        frame = require_valid_frame(
            frame,
            symbol=normalised_symbol,
            min_rows=self._config.min_history_rows,
        )
        self._cache.set(cache_key, frame)
        return frame.copy()

    def get_daily_history(
        self,
        symbol: str,
        *,
        period: str = "1mo",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Convenience wrapper for daily OHLCV data."""

        return self.get_history(symbol, period=period, interval="1d", use_cache=use_cache)

    def get_intraday_history(
        self,
        symbol: str,
        *,
        period: str = "1d",
        interval: str = "5m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Convenience wrapper for intraday OHLCV data.

        Raises :class:`MarketDataError` when the provider does not
        support intraday data for the requested ``period``/``interval``
        combination.
        """

        if not self._provider_supports_intraday(interval):
            raise MarketDataError(
                f"Provider '{self._provider.name}' does not support intraday interval '{interval}'"
            )
        return self.get_history(symbol, period=period, interval=interval, use_cache=use_cache)

    def get_quote(self, symbol: str, *, use_cache: bool = True) -> Quote:
        """Return the latest quote for ``symbol``.

        Returns a :class:`Quote` with ``is_available=False`` if the
        provider has no observation. The cache short-circuits repeat
        lookups so a single dashboard render issues at most one
        upstream call per ticker.
        """

        try:
            normalised = validate_symbol(symbol)
        except ValidationError as exc:
            raise InvalidSymbolError(str(exc)) from exc

        cache_key = f"quote:{normalised}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, Quote):
                logger.debug("Cache hit for %s", cache_key)
                return Quote(**{**cached.__dict__})

        logger.info("Fetching quote for %s (provider=%s)", normalised, self._provider.name)
        try:
            quote = self._provider.get_quote(normalised)
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected quote error for %s: %s", normalised, exc)
            raise MarketDataError(f"Failed to fetch quote for '{normalised}'") from exc

        if not isinstance(quote, Quote):
            quote = Quote.unavailable(normalised, provider=self._provider.name)
        if not quote.provider:
            quote = Quote(**{**quote.__dict__, "provider": self._provider.name})

        self._cache.set(cache_key, quote)
        return Quote(**{**quote.__dict__})

    def get_previous_close(self, symbol: str, *, use_cache: bool = True) -> float | None:
        """Return the previous close for ``symbol`` or ``None``."""

        try:
            frame = self.get_daily_history(symbol, period="5d", use_cache=use_cache)
        except (SymbolNotFoundError, InvalidSymbolError):
            return None
        return safe_previous_close(frame)

    def get_volume(self, symbol: str, *, use_cache: bool = True) -> int:
        """Return the most recent traded volume for ``symbol``."""

        try:
            frame = self.get_daily_history(symbol, period="5d", use_cache=use_cache)
        except (SymbolNotFoundError, InvalidSymbolError):
            return 0
        if frame.empty or "Volume" not in frame.columns:
            return 0
        try:
            value = int(frame["Volume"].iloc[-1])
        except (TypeError, ValueError):
            return 0
        return max(value, 0)

    def get_daily_change(self, symbol: str, *, use_cache: bool = True) -> dict[str, float | None]:
        """Return the latest price, previous close, change and percent change.

        The return value is always a dictionary so the UI can decide
        what to render. ``None`` values signal "not available" rather
        than zeroed-out numbers.
        """

        try:
            quote = self.get_quote(symbol, use_cache=use_cache)
        except (SymbolNotFoundError, InvalidSymbolError):
            return {"price": None, "previous_close": None, "change": 0.0, "change_pct": 0.0}

        if not quote.is_available:
            return {
                "price": None,
                "previous_close": quote.previous_close,
                "change": 0.0,
                "change_pct": 0.0,
            }
        return {
            "price": quote.price,
            "previous_close": quote.previous_close,
            "change": quote.change,
            "change_pct": quote.change_pct,
        }

    def get_stats(self, symbol: str, *, use_cache: bool = True) -> MarketStats:
        """Return high level market statistics for ``symbol``."""

        try:
            normalised = validate_symbol(symbol)
        except ValidationError as exc:
            raise InvalidSymbolError(str(exc)) from exc

        cache_key = f"stats:{normalised}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, MarketStats):
                logger.debug("Cache hit for %s", cache_key)
                return MarketStats(**{**cached.__dict__})

        logger.info("Fetching stats for %s (provider=%s)", normalised, self._provider.name)
        try:
            stats = self._provider.get_stats(normalised)
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected stats error for %s: %s", normalised, exc)
            raise MarketDataError(f"Failed to fetch stats for '{normalised}'") from exc

        if not isinstance(stats, MarketStats):
            stats = MarketStats.unavailable(normalised, provider=self._provider.name)
        if not stats.provider:
            stats = MarketStats(**{**stats.__dict__, "provider": self._provider.name})

        self._cache.set(cache_key, stats)
        return MarketStats(**{**stats.__dict__})

    def supports_intraday(self) -> bool:
        """Return ``True`` when the provider supports intraday intervals."""

        return self._provider_supports_intraday("5m")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_symbol_key(symbol: str) -> str | None:
        try:
            return validate_symbol(symbol)
        except ValidationError:
            return None

    def _validate_request(
        self,
        symbol: str,
        period: str | None,
        interval: str | None,
    ) -> tuple[str, str, str]:
        try:
            normalised_symbol = validate_symbol(symbol)
        except ValidationError as exc:
            raise InvalidSymbolError(str(exc)) from exc

        try:
            normalised_period = validate_period(period or CONFIG.default_period)
        except ValidationError as exc:
            raise MarketDataError(str(exc)) from exc

        try:
            normalised_interval = validate_interval(interval or CONFIG.default_interval)
        except ValidationError as exc:
            raise MarketDataError(str(exc)) from exc

        return normalised_symbol, normalised_period, normalised_interval

    def _provider_supports_intraday(self, interval: str) -> bool:
        # The yfinance provider supports all intervals declared in the
        # configuration. Other providers may override this hook. We
        # use the validation helper as a single source of truth.
        try:
            validate_interval(interval)
        except ValidationError:
            return False
        # yfinance does not return data for < 1 minute or > 90 days
        # intraday; treat the most common intraday intervals as
        # supported and let the provider raise if the combination is
        # not actually available.
        return interval.endswith("m") or interval.endswith("h")


# ---------------------------------------------------------------------------
# Process wide singleton helpers
# ---------------------------------------------------------------------------


_DEFAULT_SERVICE: MarketDataService | None = None


def get_default_service() -> MarketDataService:
    """Return a lazily constructed process-wide service instance."""

    global _DEFAULT_SERVICE  # noqa: PLW0603
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = MarketDataService()
    return _DEFAULT_SERVICE


def reset_default_service() -> None:
    """Drop the process-wide service instance (test helper)."""

    global _DEFAULT_SERVICE  # noqa: PLW0603
    _DEFAULT_SERVICE = None


# ---------------------------------------------------------------------------
# Convenience helpers used by the UI
# ---------------------------------------------------------------------------


def get_current_price(
    symbol: str, *, service: MarketDataService | None = None
) -> float | None:
    """Return the most recent price for ``symbol`` or ``None``."""

    svc = service or get_default_service()
    try:
        quote = svc.get_quote(symbol)
    except (InvalidSymbolError, MarketDataError) as exc:
        logger.warning("Unable to fetch current price for %s: %s", symbol, exc)
        return None
    return quote.price if quote.is_available else None


def get_change_percent(
    symbol: str, *, service: MarketDataService | None = None
) -> float | None:
    """Return the percentage change for ``symbol`` since the previous close."""

    svc = service or get_default_service()
    try:
        quote = svc.get_quote(symbol)
    except (InvalidSymbolError, MarketDataError) as exc:
        logger.warning("Unable to fetch change percent for %s: %s", symbol, exc)
        return None
    if not quote.is_available or quote.previous_close in (None, 0):
        return None
    if math.isnan(quote.change_pct):
        return None
    return quote.change_pct


def summarise_observation(
    symbol: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Return a dictionary summarising the most recent observation in ``frame``.

    The function is provider agnostic and is reused by the Streamlit
    dashboard. It does not raise on empty frames: it returns
    ``{"available": False, ...}`` instead.
    """

    result: dict[str, Any] = {
        "available": False,
        "price": None,
        "previous_close": None,
        "change": 0.0,
        "change_pct": 0.0,
        "volume": 0,
        "is_delayed": True,
        "timestamp": datetime.now(tz=timezone.utc),
    }
    if frame is None or frame.empty or "Close" not in frame.columns:
        return result

    price = safe_last_close(frame)
    if price is None:
        return result
    previous = safe_previous_close(frame)
    volume = 0
    if "Volume" in frame.columns:
        try:
            volume = max(int(frame["Volume"].iloc[-1]), 0)
        except (TypeError, ValueError):
            volume = 0

    change = (price - previous) if previous is not None else 0.0
    change_pct = (change / previous * 100.0) if previous else 0.0

    timestamp: Any = frame.index[-1]
    try:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        timestamp = ts.to_pydatetime()
    except (TypeError, ValueError):
        timestamp = datetime.now(tz=timezone.utc)

    result.update(
        {
            "available": True,
            "price": price,
            "previous_close": previous,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "is_delayed": True,
            "timestamp": timestamp,
        }
    )
    return result


def make_cache_key(*parts: Any) -> str:
    """Return a deterministic cache key for ``parts``.

    The helper exists for tests and for providers that want to share
    the service cache. It uses :func:`hashlib.sha256` so the result is
    stable across processes.
    """

    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
