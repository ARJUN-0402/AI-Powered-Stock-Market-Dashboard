"""News data provider implementations."""

from __future__ import annotations

from src.data.news_service.providers.base import NewsProvider
from src.data.news_service.providers.newsapi_provider import NewsAPIProvider
from src.data.news_service.providers.yfinance_news import YFinanceNewsProvider

__all__ = ["NewsProvider", "NewsAPIProvider", "YFinanceNewsProvider"]
