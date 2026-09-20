"""Abstract news provider interface.

Any concrete news source (yfinance, NewsAPI, Finnhub, ...) must implement
:class:`NewsProvider`. The interface is deliberately narrow: providers are
not aware of caching, normalisation, deduplication, sentiment or error
translation — those concerns live in
:class:`src.data.news_service.service.NewsDataService`.

Implementations should:

* Translate their native response into :class:`NewsArticle` DTOs.
* Raise :class:`NewsDataError` subclasses for known failure modes.
* Return an empty list when the upstream call succeeds but no articles are
  available.
* Never fabricate or randomise headlines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.data.news_service.dto import NewsArticle


class NewsProvider(ABC):
    """Provider-agnostic interface for fetching ticker news."""

    name: str = "abstract"

    @abstractmethod
    def fetch_news(self, symbol: str, *, limit: int = 20) -> Sequence[NewsArticle]:
        """Return normalised news articles for ``symbol``.

        Implementations must return an empty sequence when the upstream call
        succeeds but no articles are available.
        """

    def supports_symbol(self, symbol: str) -> bool:
        """Return ``True`` when the provider can resolve ``symbol``."""

        return True
