"""News data service package.

This package provides a reusable, provider-agnostic news ingestion and
financial-sentiment layer. It mirrors the architecture of
:mod:`src.data.market_data_service`:

* :class:`NewsProvider` — provider interface to implement when integrating a
  new data source.
* :class:`YFinanceNewsProvider` — the default implementation, backed by
  yfinance's real news endpoint (no credentials required).
* :class:`NewsAPIProvider` — an optional provider backed by NewsAPI.org
  (requires ``NEWSAPI_KEY``).
* :class:`NewsDataService` — the entry point used by the application. It
  validates symbols, invokes the provider, normalises / deduplicates /
  time-zone-validates articles, classifies them with a
  :class:`~src.nlp.sentiment_model.SentimentModel` (FinBERT, with a finance
  lexicon fallback) and aggregates results into daily / weekly / recent
  windows.
* :class:`NewsArticle`, :class:`ArticleSentiment`,
  :class:`AggregateSentiment` — value objects returned to callers.
* :class:`NewsDataError` and its subclasses — typed error hierarchy.
"""

from __future__ import annotations

from src.data.news_service.dto import (
    AggregateSentiment,
    ArticleSentiment,
    NewsArticle,
    ProviderStatus,
)
from src.data.news_service.errors import (
    InvalidSymbolError,
    NewsAuthenticationError,
    NewsDataError,
    NewsProviderError,
    RateLimitedError,
    TimeoutExceededError,
)
from src.data.news_service.providers.base import NewsProvider
from src.data.news_service.providers.newsapi_provider import NewsAPIProvider
from src.data.news_service.providers.yfinance_news import YFinanceNewsProvider
from src.data.news_service.service import (
    NewsDataService,
    get_default_service,
    reset_default_service,
    resolve_provider,
)

__all__ = [
    "NewsDataService",
    "NewsProvider",
    "YFinanceNewsProvider",
    "NewsAPIProvider",
    "NewsArticle",
    "ArticleSentiment",
    "AggregateSentiment",
    "ProviderStatus",
    "NewsDataError",
    "InvalidSymbolError",
    "NewsAuthenticationError",
    "NewsProviderError",
    "RateLimitedError",
    "TimeoutExceededError",
    "get_default_service",
    "reset_default_service",
    "resolve_provider",
]
