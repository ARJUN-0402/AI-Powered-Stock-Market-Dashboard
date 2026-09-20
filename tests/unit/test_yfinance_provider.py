"""Tests for the yfinance provider adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.market_data_service.errors import (
    ProviderError,
    RateLimitedError,
    TimeoutExceededError,
)
from src.data.market_data_service.providers.yfinance_provider import (
    YFinanceProvider,
    _flatten_columns,
    _looks_like_rate_limit,
    _normalise_history_frame,
)


def test_flatten_columns_handles_multiindex() -> None:
    idx = pd.MultiIndex.from_tuples([("Open",), ("Close",)], names=[None])
    frame = pd.DataFrame([[1, 2]], columns=idx)
    flat = _flatten_columns(frame)
    assert list(flat.columns) == ["Open", "Close"]


def test_normalise_handles_lowercase_columns() -> None:
    raw = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 200],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    out = _normalise_history_frame(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalise_returns_empty_for_empty_input() -> None:
    out = _normalise_history_frame(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalise_returns_empty_for_none() -> None:
    out = _normalise_history_frame(None)  # type: ignore[arg-type]
    assert out.empty


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit exceeded",
        "429 Too Many Requests",
        "Yfinance rate-limited",
    ],
)
def test_looks_like_rate_limit(message: str) -> None:
    assert _looks_like_rate_limit(message) is True


def test_safe_call_translates_timeout() -> None:
    from src.data.market_data_service.providers.yfinance_provider import _safe_call

    def _raise() -> None:
        raise TimeoutError("slow")

    with pytest.raises(TimeoutExceededError):
        _safe_call(_raise, timeout=0.1, label="history")


def test_safe_call_translates_rate_limit() -> None:
    from src.data.market_data_service.providers.yfinance_provider import _safe_call

    def _raise() -> None:
        raise RuntimeError("429 too many requests")

    with pytest.raises(RateLimitedError):
        _safe_call(_raise, timeout=0.1, label="history")


def test_safe_call_translates_generic_error() -> None:
    from src.data.market_data_service.providers.yfinance_provider import _safe_call

    def _raise() -> None:
        raise RuntimeError("unexpected")

    with pytest.raises(ProviderError):
        _safe_call(_raise, timeout=0.1, label="history")


def test_get_quote_returns_unavailable_for_empty_history() -> None:
    class _Stub(YFinanceProvider):
        def __init__(self) -> None:
            super().__init__(timeout=0.1)

        def get_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:  # type: ignore[override]
            return pd.DataFrame()

    stub = _Stub()
    quote = stub.get_quote("AAPL")
    assert quote.is_available is False
    assert quote.is_delayed is True
