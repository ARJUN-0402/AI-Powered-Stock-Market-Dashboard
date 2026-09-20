"""News data layer (backward-compatible facade).

This module is a thin shim over :class:`src.data.news_service.NewsDataService`.
It preserves the historical :func:`fetch_news` entry point for legacy callers
but now returns real, normalized :class:`NewsArticle` objects instead of the
previous synthetic headlines. New code should call the service directly:

    from src.data.news_service import get_default_service
    service = get_default_service()
    articles = service.fetch_articles(symbol)
    scored = service.get_news_with_sentiment(symbol)
    aggregate = service.get_aggregate_sentiment(symbol)

The functions here never raise: provider failures are logged and surfaced as
an empty list so the UI can degrade gracefully.
"""

from __future__ import annotations

from src.data.news_service import NewsArticle, get_default_service
from src.data.news_service.service import NewsDataService
from src.utils.logging import get_logger
from src.utils.validation import validate_symbol

logger = get_logger(__name__)

__all__ = ["fetch_news", "fetch_news_with_sentiment", "get_news_service", "NewsArticle"]


def get_news_service() -> NewsDataService:
    """Return the process-wide :class:`NewsDataService` instance."""

    return get_default_service()


def fetch_news(symbol: str) -> list[NewsArticle]:
    """Return real, normalized news articles for ``symbol``.

    Never raises: on validation or provider failure the error is logged and
    an empty list is returned so callers (notably the dashboard) can render
    an honest "no news" state.
    """

    try:
        normalised = validate_symbol(symbol)
    except Exception as exc:  # noqa: BLE001 - validation guard
        logger.warning("Invalid symbol for news fetch: %s", exc)
        return []

    try:
        return get_news_service().fetch_articles(normalised)
    except Exception as exc:  # noqa: BLE001 - provider guard
        logger.exception("Failed to fetch news for %s: %s", normalised, exc)
        return []


def fetch_news_with_sentiment(symbol: str) -> list:
    """Return articles enriched with sentiment scores for ``symbol``.

    Never raises; returns an empty list on failure.
    """

    try:
        normalised = validate_symbol(symbol)
    except Exception as exc:  # noqa: BLE001 - validation guard
        logger.warning("Invalid symbol for news fetch: %s", exc)
        return []
    try:
        return get_news_service().get_news_with_sentiment(normalised)
    except Exception as exc:  # noqa: BLE001 - provider guard
        logger.exception("Failed to fetch news sentiment for %s: %s", normalised, exc)
        return []
