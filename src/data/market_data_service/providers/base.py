"""Abstract market data provider.

Any concrete provider (yfinance, Alpha Vantage, Polygon, IEX, ...)
must implement :class:`MarketDataProvider`. The interface is
intentionally narrow: providers are not aware of caching, validation
or error translation — those concerns live in
:class:`src.data.market_data_service.service.MarketDataService`.

Implementations should:

* Be side effect free apart from the upstream API call.
* Raise :class:`MarketDataError` subclasses for known failure modes
  (the service will translate provider-specific exceptions into the
  appropriate subclass).
* Return :class:`Quote` / :class:`MarketStats` with
  ``is_available=False`` when the upstream call succeeds but returns
  no usable data, rather than raising.
* Never fabricate or randomise data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.data.market_data_service.dto import MarketStats, Quote


class MarketDataProvider(ABC):
    """Provider-agnostic interface for fetching market data."""

    name: str = "abstract"

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        """Return a validated OHLCV frame for ``symbol``.

        Implementations must return an empty
        :class:`pandas.DataFrame` (with the expected columns) when the
        upstream call succeeds but no rows are available. The service
        is responsible for translating empty responses into a
        :class:`SymbolNotFoundError` if appropriate.
        """

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return the latest available quote for ``symbol``."""

    @abstractmethod
    def get_stats(self, symbol: str) -> MarketStats:
        """Return high level statistics for ``symbol``."""
