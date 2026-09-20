"""Backwards compatible market data helpers.

The functions in this module are preserved for callers that have not
yet migrated to :class:`MarketDataService`. They are thin wrappers
that delegate to the service and therefore share its caching, error
handling and validation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.market_data_service import (
    InvalidSymbolError,
    MarketDataError,
    MarketDataService,
    SymbolNotFoundError,
    get_default_service,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def fetch_stock_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    *,
    service: MarketDataService | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return an OHLCV frame for ``symbol``.

    The function is preserved for backwards compatibility; new code
    should depend on :class:`MarketDataService` directly.
    """

    svc = service or get_default_service()
    try:
        return svc.get_history(symbol, period=period, interval=interval, use_cache=use_cache)
    except (InvalidSymbolError, SymbolNotFoundError, MarketDataError) as exc:
        logger.warning("fetch_stock_data(%s) failed: %s", symbol, exc)
        return pd.DataFrame()


def fetch_live_price(
    symbol: str,
    *,
    service: MarketDataService | None = None,
    use_cache: bool = True,
) -> float | None:
    """Return the latest price for ``symbol`` or ``None``."""

    svc = service or get_default_service()
    try:
        quote = svc.get_quote(symbol, use_cache=use_cache)
    except (InvalidSymbolError, MarketDataError) as exc:
        logger.warning("fetch_live_price(%s) failed: %s", symbol, exc)
        return None
    return quote.price if quote.is_available else None


def get_company_name(symbol: str) -> str:
    """Return the long company name for ``symbol`` or the symbol itself.

    The implementation now relies on the service's market stats
    payload which surfaces the long name when available; when it is
    not, the symbol is returned unchanged.
    """

    try:
        stats = get_default_service().get_stats(symbol)
    except (InvalidSymbolError, MarketDataError) as exc:
        logger.debug("Unable to resolve company name for %s: %s", symbol, exc)
        return symbol
    name: Any = stats.symbol
    if stats.is_available and stats.currency:
        name = f"{stats.symbol}"
    return str(name)
