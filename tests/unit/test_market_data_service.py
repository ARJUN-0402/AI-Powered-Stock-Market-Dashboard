"""Tests for the centralised market data service."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.market_data_service import (
    InvalidSymbolError,
    MarketDataError,
    MarketDataService,
    MarketStats,
    ProviderError,
    Quote,
    RateLimitedError,
    SymbolNotFoundError,
    YFinanceProvider,
)
from src.data.market_data_service.service import (
    MarketDataServiceConfig,
    get_change_percent,
    get_current_price,
    summarise_observation,
)
from tests.fixtures import make_price_frame

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider(YFinanceProvider):
    """Provider that records calls and returns a fixed frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__(timeout=0.1)
        self._frame = frame
        self.history_calls: list[tuple[str, str, str]] = []
        self.quote_calls: list[str] = []
        self.stats_calls: list[str] = []

    def get_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:  # type: ignore[override]
        self.history_calls.append((symbol, period, interval))
        return self._frame

    def get_quote(self, symbol: str) -> Quote:  # type: ignore[override]
        self.quote_calls.append(symbol)
        if self._frame.empty:
            return Quote.unavailable(symbol, provider=self.name)
        last = float(self._frame["Close"].iloc[-1])
        previous = (
            float(self._frame["Close"].iloc[-2]) if len(self._frame) >= 2 else None
        )
        change = last - previous if previous is not None else 0.0
        change_pct = (change / previous * 100.0) if previous else 0.0
        timestamp = self._frame.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            ts = timestamp.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = datetime.now(tz=timezone.utc)
        return Quote(
            symbol=symbol,
            price=last,
            previous_close=previous,
            change=change,
            change_pct=change_pct,
            volume=int(self._frame["Volume"].iloc[-1]),
            timestamp=ts,
            is_delayed=True,
            is_available=True,
            provider=self.name,
        )

    def get_stats(self, symbol: str) -> MarketStats:  # type: ignore[override]
        self.stats_calls.append(symbol)
        return MarketStats(
            symbol=symbol,
            open_price=float(self._frame["Open"].iloc[-1]),
            high_price=float(self._frame["High"].iloc[-1]),
            low_price=float(self._frame["Low"].iloc[-1]),
            close_price=float(self._frame["Close"].iloc[-1]),
            previous_close=float(self._frame["Close"].iloc[-2])
            if len(self._frame) >= 2
            else None,
            volume=int(self._frame["Volume"].iloc[-1]),
            is_delayed=True,
            is_available=True,
            provider=self.name,
        )


class _ExplodingProvider(YFinanceProvider):
    def __init__(self, exc: BaseException) -> None:
        super().__init__(timeout=0.1)
        self._exc = exc

    def get_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:  # type: ignore[override]
        raise self._exc

    def get_quote(self, symbol: str) -> Quote:  # type: ignore[override]
        raise self._exc

    def get_stats(self, symbol: str) -> MarketStats:  # type: ignore[override]
        raise self._exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_symbol_raises() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(10)))
    with pytest.raises(InvalidSymbolError):
        service.get_history("BAD!")


def test_empty_symbol_raises() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(10)))
    with pytest.raises(InvalidSymbolError):
        service.get_history("")


def test_invalid_period_raises() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(10)))
    with pytest.raises(MarketDataError):
        service.get_history("AAPL", period="bogus")


def test_invalid_interval_raises() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(10)))
    with pytest.raises(MarketDataError):
        service.get_history("AAPL", interval="bogus")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_get_history_returns_frame_for_valid_symbol() -> None:
    frame = make_price_frame(rows=30)
    provider = _FakeProvider(frame)
    service = MarketDataService(provider=provider)
    result = service.get_history("AAPL", period="1mo", interval="1d")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert provider.history_calls == [("AAPL", "1mo", "1d")]


def test_get_quote_returns_quote_with_change() -> None:
    frame = make_price_frame(rows=10)
    service = MarketDataService(provider=_FakeProvider(frame))
    quote = service.get_quote("AAPL")
    assert quote.is_available
    assert quote.price is not None
    assert quote.previous_close is not None
    assert quote.is_delayed
    assert quote.timestamp.tzinfo is not None
    assert quote.change == pytest.approx(quote.price - quote.previous_close)


def test_get_stats_returns_stats() -> None:
    frame = make_price_frame(rows=10)
    service = MarketDataService(provider=_FakeProvider(frame))
    stats = service.get_stats("AAPL")
    assert stats.is_available
    assert stats.close_price is not None
    assert stats.is_delayed


def test_daily_change_helper() -> None:
    frame = make_price_frame(rows=10)
    service = MarketDataService(provider=_FakeProvider(frame))
    snapshot = service.get_daily_change("AAPL")
    assert snapshot["price"] is not None
    assert snapshot["previous_close"] is not None


# ---------------------------------------------------------------------------
# Empty / malformed responses
# ---------------------------------------------------------------------------


def test_empty_dataset_raises_symbol_not_found() -> None:
    service = MarketDataService(provider=_FakeProvider(pd.DataFrame()))
    with pytest.raises(SymbolNotFoundError):
        service.get_history("AAPL", period="1mo")


def test_missing_columns_raises_market_data_error() -> None:
    bad = make_price_frame(rows=30).drop(columns=["Close"])
    service = MarketDataService(provider=_FakeProvider(bad))
    with pytest.raises(MarketDataError):
        service.get_history("AAPL", period="1mo")


def test_insufficient_rows_raises_symbol_not_found() -> None:
    small = make_price_frame(rows=1)
    service = MarketDataService(
        provider=_FakeProvider(small),
        config=MarketDataServiceConfig(min_history_rows=10),
    )
    with pytest.raises(SymbolNotFoundError):
        service.get_history("AAPL", period="1mo")


# ---------------------------------------------------------------------------
# Provider exceptions
# ---------------------------------------------------------------------------


def test_provider_exception_translated_to_market_data_error() -> None:
    service = MarketDataService(
        provider=_ExplodingProvider(ProviderError("boom")),
    )
    with pytest.raises(ProviderError):
        service.get_history("AAPL", period="1mo")


def test_rate_limited_exception_propagates() -> None:
    service = MarketDataService(
        provider=_ExplodingProvider(RateLimitedError("throttled")),
    )
    with pytest.raises(RateLimitedError):
        service.get_history("AAPL", period="1mo")


def test_unexpected_exception_wrapped() -> None:
    class _BoomProvider(_FakeProvider):
        def get_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:  # type: ignore[override]
            raise RuntimeError("nope")

    service = MarketDataService(provider=_BoomProvider(make_price_frame(10)))
    with pytest.raises(MarketDataError):
        service.get_history("AAPL", period="1mo")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cached_history_avoids_repeat_provider_call() -> None:
    provider = _FakeProvider(make_price_frame(rows=30))
    service = MarketDataService(provider=provider)
    first = service.get_history("AAPL", period="1mo")
    second = service.get_history("AAPL", period="1mo")
    assert first.equals(second)
    assert len(provider.history_calls) == 1


def test_caching_can_be_bypassed() -> None:
    provider = _FakeProvider(make_price_frame(rows=30))
    service = MarketDataService(provider=provider)
    service.get_history("AAPL", period="1mo", use_cache=False)
    service.get_history("AAPL", period="1mo", use_cache=False)
    assert len(provider.history_calls) == 2


def test_caching_distinguishes_parameters() -> None:
    provider = _FakeProvider(make_price_frame(rows=30))
    service = MarketDataService(provider=provider)
    service.get_history("AAPL", period="1mo")
    service.get_history("AAPL", period="3mo")
    assert len(provider.history_calls) == 2


def test_refresh_invalidates_specific_symbol() -> None:
    provider = _FakeProvider(make_price_frame(rows=30))
    service = MarketDataService(provider=provider)
    service.get_history("AAPL", period="1mo")
    service.refresh("AAPL")
    service.get_history("AAPL", period="1mo")
    assert len(provider.history_calls) == 2


def test_refresh_clears_cache() -> None:
    provider = _FakeProvider(make_price_frame(rows=30))
    service = MarketDataService(provider=provider)
    service.get_history("AAPL", period="1mo")
    service.refresh()
    service.get_history("AAPL", period="1mo")
    assert len(provider.history_calls) == 2


def test_quote_is_cached() -> None:
    provider = _FakeProvider(make_price_frame(rows=30))
    service = MarketDataService(provider=provider)
    service.get_quote("AAPL")
    service.get_quote("AAPL")
    assert len(provider.quote_calls) == 1


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def test_get_current_price_returns_float() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    price = get_current_price("AAPL", service=service)
    assert isinstance(price, float)
    assert price > 0


def test_get_change_percent_returns_float() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    pct = get_change_percent("AAPL", service=service)
    assert isinstance(pct, float)


def test_summarise_observation_handles_empty_frame() -> None:
    summary = summarise_observation("AAPL", pd.DataFrame())
    assert summary["available"] is False
    assert summary["price"] is None


def test_summarise_observation_returns_metadata() -> None:
    frame = make_price_frame(rows=10)
    summary = summarise_observation("AAPL", frame)
    assert summary["available"] is True
    assert summary["price"] is not None
    assert summary["is_delayed"] is True


def test_get_previous_close_helper() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    previous = service.get_previous_close("AAPL")
    assert previous is not None


def test_get_volume_helper() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    volume = service.get_volume("AAPL")
    assert volume >= 0


# ---------------------------------------------------------------------------
# Quote helper
# ---------------------------------------------------------------------------


def test_get_quote_returns_unavailable_for_empty_frame() -> None:
    service = MarketDataService(provider=_FakeProvider(pd.DataFrame()))
    quote = service.get_quote("AAPL")
    assert quote.is_available is False
    assert quote.is_delayed is True


# ---------------------------------------------------------------------------
# Configure / provider
# ---------------------------------------------------------------------------


def test_provider_name_is_exposed() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    assert service.provider_name == "yfinance"


def test_configure_updates_ttls() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    service.configure(history_ttl_seconds=123, negative_ttl_seconds=10)
    assert service.cache_stats["size"] >= 0


def test_intraday_uses_provider_path() -> None:
    provider = _FakeProvider(make_price_frame(rows=80))
    service = MarketDataService(provider=provider)
    result = service.get_intraday_history("AAPL", period="1d", interval="5m")
    assert not result.empty


def test_get_quote_invalid_symbol_raises() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    with pytest.raises(InvalidSymbolError):
        service.get_quote("@@")


def test_get_stats_invalid_symbol_raises() -> None:
    service = MarketDataService(provider=_FakeProvider(make_price_frame(rows=10)))
    with pytest.raises(InvalidSymbolError):
        service.get_stats("@@")
