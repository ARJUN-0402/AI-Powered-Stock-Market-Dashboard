"""Lightweight portfolio analysis helpers.

This module offers pure functions for summarising a watchlist of holdings.
It deliberately avoids any persistence layer so it remains easy to test.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from src.utils.validation import safe_float


@dataclass(frozen=True)
class Holding:
    """A single position summary."""

    symbol: str
    current_price: float
    previous_price: float

    @property
    def change(self) -> float:
        return self.current_price - self.previous_price

    @property
    def change_pct(self) -> float:
        if self.previous_price == 0:
            return 0.0
        return (self.change / self.previous_price) * 100.0


def summarise_price_frame(symbol: str, data: pd.DataFrame) -> Holding:
    """Return a :class:`Holding` summarising the latest two rows of ``data``."""

    if data is None or data.empty or "Close" not in data.columns:
        return Holding(symbol=symbol, current_price=0.0, previous_price=0.0)

    current = safe_float(data["Close"].iloc[-1])
    previous = safe_float(data["Close"].iloc[-2]) if len(data) > 1 else current
    return Holding(symbol=symbol, current_price=current, previous_price=previous)


def summarise_holdings(items: Iterable[Holding]) -> list[Holding]:
    """Return a list of :class:`Holding` instances, preserving order."""

    return list(items)


def total_market_value(holdings: Iterable[Holding]) -> float:
    """Return the sum of current prices across ``holdings``."""

    return float(sum(h.current_price for h in holdings))


def top_movers(holdings: Iterable[Holding], n: int = 3) -> list[Holding]:
    """Return the top ``n`` movers sorted by absolute percentage change."""

    ordered = sorted(holdings, key=lambda h: abs(h.change_pct), reverse=True)
    return ordered[:n]
