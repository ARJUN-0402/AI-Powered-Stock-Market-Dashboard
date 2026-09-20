"""Market data service package.

This package provides a reusable, provider-agnostic market data layer
suitable for the Streamlit dashboard and any future analytical
consumers (notebooks, scheduled jobs, REST APIs, ...).

Public surface:

* :class:`MarketDataService` — the entry point used by the application.
* :class:`MarketDataProvider` — provider interface to implement when
  integrating a new data source.
* :class:`YFinanceProvider` — the default implementation backed by
  :mod:`yfinance`.
* :class:`Quote` and :class:`MarketStats` — value objects returned to
  callers.
* :class:`MarketDataError` and its subclasses — typed error hierarchy.
"""

from __future__ import annotations

from src.data.market_data_service.dto import MarketStats, Quote
from src.data.market_data_service.errors import (
    InvalidSymbolError,
    MarketDataError,
    ProviderError,
    RateLimitedError,
    SymbolNotFoundError,
    TimeoutExceededError,
)
from src.data.market_data_service.providers.base import MarketDataProvider
from src.data.market_data_service.providers.yfinance_provider import YFinanceProvider
from src.data.market_data_service.service import (
    MarketDataService,
    get_current_price,
    get_default_service,
    reset_default_service,
)

__all__ = [
    "MarketDataError",
    "InvalidSymbolError",
    "SymbolNotFoundError",
    "ProviderError",
    "RateLimitedError",
    "TimeoutExceededError",
    "MarketDataProvider",
    "YFinanceProvider",
    "MarketDataService",
    "Quote",
    "MarketStats",
    "get_default_service",
    "get_current_price",
    "reset_default_service",
]
