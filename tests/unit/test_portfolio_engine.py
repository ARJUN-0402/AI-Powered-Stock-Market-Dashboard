"""Tests for the portfolio engine."""

from __future__ import annotations

from src.portfolio.portfolio_engine import (
    Holding,
    summarise_holdings,
    summarise_price_frame,
    top_movers,
    total_market_value,
)
from tests.fixtures import make_price_frame


def test_summarise_price_frame_returns_holding() -> None:
    data = make_price_frame(rows=10)
    holding = summarise_price_frame("AAPL", data)
    assert isinstance(holding, Holding)
    assert holding.symbol == "AAPL"
    assert holding.current_price > 0


def test_summarise_price_frame_handles_empty() -> None:
    import pandas as pd

    holding = summarise_price_frame("AAPL", pd.DataFrame())
    assert holding.current_price == 0.0
    assert holding.previous_price == 0.0
    assert holding.change_pct == 0.0


def test_total_market_value_sums_current_prices() -> None:
    holdings = [
        Holding(symbol="A", current_price=10.0, previous_price=9.0),
        Holding(symbol="B", current_price=20.0, previous_price=22.0),
    ]
    assert total_market_value(holdings) == 30.0


def test_summarise_holdings_preserves_order() -> None:
    holdings = [
        Holding(symbol="A", current_price=10.0, previous_price=9.0),
        Holding(symbol="B", current_price=20.0, previous_price=22.0),
    ]
    result = summarise_holdings(holdings)
    assert [h.symbol for h in result] == ["A", "B"]


def test_top_movers_returns_largest_absolute_change() -> None:
    holdings = [
        Holding(symbol="A", current_price=10.0, previous_price=9.0),  # ~11%
        Holding(symbol="B", current_price=20.0, previous_price=15.0),  # 33%
        Holding(symbol="C", current_price=5.0, previous_price=4.9),  # 2%
    ]
    movers = top_movers(holdings, n=2)
    assert [h.symbol for h in movers] == ["B", "A"]
