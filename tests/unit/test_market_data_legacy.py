"""Tests for the backwards compatible market data shim."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.market_data import fetch_live_price, fetch_stock_data
from src.data.market_data_service import MarketDataService
from src.data.market_data_service.service import (
    MarketDataServiceConfig,
    reset_default_service,
)
from tests.fixtures import make_price_frame


class _FakeProvider:
    name = "fake"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        return self._frame

    def get_quote(self, symbol: str):  # type: ignore[no-untyped-def]
        from src.data.market_data_service.dto import Quote

        if self._frame.empty:
            return Quote.unavailable(symbol, provider=self.name)
        return Quote(
            symbol=symbol,
            price=float(self._frame["Close"].iloc[-1]),
            previous_close=(
                float(self._frame["Close"].iloc[-2]) if len(self._frame) >= 2 else None
            ),
            change=0.0,
            change_pct=0.0,
            volume=0,
            timestamp=datetime.now(tz=timezone.utc),
            is_delayed=True,
            is_available=True,
            provider=self.name,
        )

    def get_stats(self, symbol: str):  # type: ignore[no-untyped-def]
        from src.data.market_data_service.dto import MarketStats

        return MarketStats(symbol=symbol, provider=self.name)


def setup_function(_: object) -> None:
    reset_default_service()


def teardown_function(_: object) -> None:
    reset_default_service()


def test_fetch_stock_data_uses_service() -> None:
    frame = make_price_frame(rows=20)
    service = MarketDataService(
        provider=_FakeProvider(frame),  # type: ignore[arg-type]
        config=MarketDataServiceConfig(min_history_rows=2),
    )
    result = fetch_stock_data("AAPL", period="1mo", service=service)
    assert not result.empty


def test_fetch_stock_data_returns_empty_for_invalid_symbol() -> None:
    service = MarketDataService(
        provider=_FakeProvider(pd.DataFrame()),  # type: ignore[arg-type]
        config=MarketDataServiceConfig(min_history_rows=2),
    )
    assert fetch_stock_data("@@", service=service).empty


def test_fetch_stock_data_returns_empty_for_symbol_not_found() -> None:
    service = MarketDataService(
        provider=_FakeProvider(pd.DataFrame()),  # type: ignore[arg-type]
        config=MarketDataServiceConfig(min_history_rows=2),
    )
    assert fetch_stock_data("ZZZZ", service=service).empty


def test_fetch_live_price_returns_float() -> None:
    service = MarketDataService(
        provider=_FakeProvider(make_price_frame(rows=20)),  # type: ignore[arg-type]
    )
    price = fetch_live_price("AAPL", service=service)
    assert isinstance(price, float)


def test_fetch_live_price_returns_none_for_unavailable() -> None:
    service = MarketDataService(
        provider=_FakeProvider(pd.DataFrame()),  # type: ignore[arg-type]
    )
    assert fetch_live_price("AAPL", service=service) is None


def test_fetch_live_price_returns_none_for_invalid_symbol() -> None:
    service = MarketDataService(
        provider=_FakeProvider(pd.DataFrame()),  # type: ignore[arg-type]
    )
    assert fetch_live_price("@@", service=service) is None
