"""yfinance implementation of the market data provider interface.

The adapter is responsible for:

* Translating yfinance specific calls into the provider interface.
* Normalising the column names returned by yfinance (which vary across
  versions — sometimes lower case, sometimes capitalised, sometimes
  with a ``Adj Close`` column and sometimes without).
* Detecting known failure modes and raising the appropriate
  :mod:`src.data.market_data_service.errors` subclass.
* Marking quotes as delayed: yfinance quotes are end-of-day for the
  free tier and intraday observations are delayed by at least
  15 minutes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.market_data_service.dto import MarketStats, Quote
from src.data.market_data_service.errors import (
    ProviderError,
    RateLimitedError,
    SymbolNotFoundError,
    TimeoutExceededError,
)
from src.data.market_data_service.providers.base import MarketDataProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

_RATE_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rate[\s\-]*limit", re.IGNORECASE),
    re.compile(r"too\s*many\s*requests", re.IGNORECASE),
    re.compile(r"429"),
)

# yfinance returns a MultiIndex column header on some versions when
# the ticker is passed alongside others. We always ask for a single
# ticker so a single-level frame is expected, but the helper below is
# defensive against future regressions.
_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _normalise_history_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with canonical OHLCV columns and a sorted index."""

    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(_OHLCV_COLUMNS))

    frame = _flatten_columns(raw)
    rename = {col: col.capitalize() for col in frame.columns if isinstance(col, str)}
    frame = frame.rename(columns=rename)
    for required in _OHLCV_COLUMNS:
        if required not in frame.columns:
            frame[required] = pd.NA
    frame = frame[list(_OHLCV_COLUMNS)].copy()
    frame = frame.sort_index()
    return frame


def _coerce_utc(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone.utc).to_pydatetime()


def _looks_like_rate_limit(message: str) -> bool:
    return any(pattern.search(message) for pattern in _RATE_LIMIT_PATTERNS)


def _safe_call(callable_: Any, *, timeout: float, label: str) -> Any:
    """Invoke ``callable_`` translating network failures into typed errors."""

    try:
        return callable_()
    except TimeoutError as exc:
        logger.warning("yfinance %s timed out after %.1fs", label, timeout)
        raise TimeoutExceededError(f"Timed out calling yfinance {label}") from exc
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - we translate everything
        message = str(exc)
        if _looks_like_rate_limit(message):
            logger.warning("yfinance %s rate limited: %s", label, message)
            raise RateLimitedError(f"Rate limited by yfinance during {label}") from exc
        logger.exception("yfinance %s failed: %s", label, exc)
        raise ProviderError(f"yfinance {label} failed: {message}") from exc


class YFinanceProvider(MarketDataProvider):
    """Provider backed by the :mod:`yfinance` package."""

    name = "yfinance"

    def __init__(self, *, session: Any | None = None, timeout: float = 10.0) -> None:
        self._timeout = float(timeout)
        self._session = session
        # yfinance is imported lazily so unit tests and offline runs do
        # not require it installed at import time.
        self._yf: Any | None = None

    def _get_yf(self) -> Any:
        if self._yf is None:
            import yfinance as yf  # type: ignore

            self._yf = yf
        return self._yf

    def _ticker(self, symbol: str) -> Any:
        yf = self._get_yf()
        if self._session is not None:
            return yf.Ticker(symbol, session=self._session)
        return yf.Ticker(symbol)

    def get_history(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        ticker = self._ticker(symbol)
        data = _safe_call(
            lambda: ticker.history(period=period, interval=interval),
            timeout=self._timeout,
            label="history",
        )
        return _normalise_history_frame(data)

    def get_quote(self, symbol: str) -> Quote:
        """Return the most recent quote from the last trading day.

        yfinance does not provide a true real-time feed on the free
        tier, so every observation is marked as delayed.
        """

        history = self.get_history(symbol, period="5d", interval="1d")
        if history.empty:
            return Quote.unavailable(symbol, provider=self.name)

        last_row = history.iloc[-1]
        previous_close: float | None = None
        if len(history) >= 2:
            previous_close = float(history["Close"].iloc[-2])

        close = _safe_float(last_row.get("Close"))
        if close is None:
            return Quote.unavailable(symbol, provider=self.name)

        volume = _safe_int(last_row.get("Volume"))
        timestamp = _coerce_utc(history.index[-1]) or datetime.now(tz=timezone.utc)

        change = (close - previous_close) if previous_close is not None else 0.0
        change_pct = (change / previous_close * 100.0) if previous_close else 0.0

        return Quote(
            symbol=symbol,
            price=close,
            previous_close=previous_close,
            change=change,
            change_pct=change_pct,
            volume=volume,
            timestamp=timestamp,
            is_delayed=True,
            is_available=True,
            currency=None,
            provider=self.name,
        )

    def get_stats(self, symbol: str) -> MarketStats:
        ticker = self._ticker(symbol)
        info = _safe_call(
            lambda: getattr(ticker, "info", None),
            timeout=self._timeout,
            label="info",
        )
        if not isinstance(info, dict) or not info:
            raise SymbolNotFoundError(f"No market stats available for '{symbol}'")

        def _maybe(key: str) -> float | None:
            value = info.get(key)
            return _safe_float(value) if value is not None else None

        return MarketStats(
            symbol=symbol,
            open_price=_maybe("open"),
            high_price=_maybe("dayHigh") or _maybe("regularMarketDayHigh"),
            low_price=_maybe("dayLow") or _maybe("regularMarketDayLow"),
            close_price=_maybe("previousClose") or _maybe("regularMarketPreviousClose"),
            previous_close=_maybe("previousClose") or _maybe("regularMarketPreviousClose"),
            volume=_safe_int(info.get("volume") or info.get("regularMarketVolume")) or 0,
            fifty_two_week_high=_maybe("fiftyTwoWeekHigh"),
            fifty_two_week_low=_maybe("fiftyTwoWeekLow"),
            market_cap=_safe_float(info.get("marketCap")),
            currency=info.get("currency"),
            is_delayed=True,
            is_available=True,
            provider=self.name,
            timestamp=datetime.now(tz=timezone.utc),
        )


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    import math

    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _safe_int(value: Any) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return 0
    return result if result >= 0 else 0
