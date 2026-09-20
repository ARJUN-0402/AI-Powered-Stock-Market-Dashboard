"""yfinance-backed news provider (no API key required).

This is the default provider because it returns real, ticker-specific
financial news from Yahoo Finance without requiring any credentials. The
adapter isolates the rest of the pipeline from yfinance's response shape.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import requests

from src.data.news_service.dto import NewsArticle
from src.data.news_service.errors import (
    NewsProviderError,
    RateLimitedError,
    TimeoutExceededError,
)
from src.data.news_service.providers.base import NewsProvider
from src.data.news_service.validators import (
    normalise_description,
    normalise_title,
    parse_timestamp_utc,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

_RATE_LIMIT_PATTERNS = (
    r"rate[ _-]*limit",
    r"too[ _]*many[ _]*requests",
    r"\b429\b",
    r"throttl",
)


class YFinanceNewsProvider(NewsProvider):
    """Provider backed by :mod:`yfinance`'s ``Ticker.news`` endpoint."""

    name = "yfinance"

    def __init__(self, *, timeout: float = 10.0, session: Any | None = None) -> None:
        self._timeout = float(timeout)
        self._session = session
        self._yf: Any | None = None

    def _get_yf(self) -> Any:
        if self._yf is None:
            import yfinance as yf  # type: ignore

            self._yf = yf
        return self._yf

    def _ticker(self, symbol: str) -> Any:
        yf = self._get_yf()
        if self._session is not None:
            return yf.Ticker(symbol, session=self._session)
        return yf.Ticker(symbol)

    def fetch_news(self, symbol: str, *, limit: int = 20) -> list[NewsArticle]:
        ticker = self._ticker(symbol)
        raw: Any = _safe_call(
            lambda: ticker.news,
            timeout=self._timeout,
            label="yfinance news",
        )

        if not isinstance(raw, list):
            return []

        articles: list[NewsArticle] = []
        for item in raw[:limit]:
            article = self._parse(item, symbol=symbol)
            if article is not None:
                articles.append(article)
        return articles

    @staticmethod
    def _parse(item: dict[str, Any], *, symbol: str) -> NewsArticle | None:
        if not isinstance(item, dict):
            return None
        content = item.get("content") or item.get("data") or {}
        if not isinstance(content, dict):
            content = {}

        title = normalise_title(content.get("title") or item.get("title"))
        if not title:
            return None

        description = normalise_description(content.get("description") or content.get("summary"))
        source = _extract_source(content)
        url = _extract_url(content)
        published_at = parse_timestamp_utc(content.get("pubDate") or content.get("publishedAt"))
        if published_at is None:
            published_at = parse_timestamp_utc(item.get("created"))
        if published_at is None:
            logger.debug("yfinance article missing timestamp: %s", url or title)

        return NewsArticle(
            symbol=symbol,
            title=title,
            source=source,
            url=url,
            published_at=published_at or datetime.now(),
            description=description,
            provider=YFinanceNewsProvider.name,
        )


def _extract_source(content: dict[str, Any]) -> str:
    provider = content.get("provider") or {}
    if isinstance(provider, dict):
        name = provider.get("displayName") or provider.get("name")
        if name:
            return str(name)
    return "Yahoo Finance"


def _extract_url(content: dict[str, Any]) -> str:
    canonical = content.get("canonicalUrl")
    if isinstance(canonical, dict) and canonical.get("url"):
        return str(canonical["url"])
    click = content.get("clickThroughUrl")
    if isinstance(click, dict) and click.get("url"):
        return str(click["url"])
    return ""


def _looks_like_rate_limit(message: str) -> bool:
    lowered = message.lower()
    return any(re.search(pattern, lowered) for pattern in _RATE_LIMIT_PATTERNS)


def _safe_call(callable_: Any, *, timeout: float, label: str) -> Any:
    """Invoke ``callable_`` translating network failures into typed errors."""

    try:
        return callable_()
    except TimeoutError as exc:
        logger.warning("yfinance %s timed out after %.1fs", label, timeout)
        raise TimeoutExceededError(f"Timed out calling yfinance {label}") from exc
    except (NewsProviderError, RateLimitedError, TimeoutExceededError):
        raise
    except (requests.Timeout, requests.ConnectionError) as exc:
        message = str(exc)
        if "timeout" in message.lower():
            raise TimeoutExceededError(f"yfinance {label} timed out") from exc
        if _looks_like_rate_limit(message):
            raise RateLimitedError(f"Rate limited by yfinance during {label}") from exc
        raise NewsProviderError(f"yfinance {label} failed: {message}") from exc
    except Exception as exc:  # noqa: BLE001 - we translate everything
        message = str(exc)
        if _looks_like_rate_limit(message):
            logger.warning("yfinance %s rate limited: %s", label, message)
            raise RateLimitedError(f"Rate limited by yfinance during {label}") from exc
        logger.exception("yfinance %s failed: %s", label, exc)
        raise NewsProviderError(f"yfinance {label} failed: {message}") from exc
