"""NewsAPI.org-backed news provider (requires an API key).

This provider fetches real articles from the NewsAPI Everything endpoint
using the ticker symbol as the query. Unlike the yfinance provider it does
not filter to finance-specific sources, so for market-focused analysis the
yfinance provider remains the better default; NewsAPI is useful when a
broader set of headlines is desired.

The API key is read from the ``NEWSAPI_KEY`` environment variable / Streamlit
secrets. It is **never** hard-coded. See ``.env.example`` for setup.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import requests

from src.data.news_service.dto import NewsArticle
from src.data.news_service.errors import (
    NewsAuthenticationError,
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

_ENDPOINT = "https://newsapi.org/v2/everything"
_RATE_LIMIT_PATTERNS = (
    r"rate[ _-]*limit",
    r"too[ _]*many[ _]*requests",
    r"\b429\b",
    r"throttl",
)


def resolve_api_key(key_name: str = "NEWSAPI_KEY") -> str | None:
    """Resolve a secret key from Streamlit secrets then the environment.

    Checking Streamlit first lets dashboard operators store keys in
    ``.streamlit/secrets.toml`` (which is git-ignored) while keeping the
    provider importable in headless contexts.
    """

    try:
        import streamlit as st  # type: ignore

        try:
            value = st.secrets.get(key_name)
            if isinstance(value, str) and value:
                return value
        except Exception:  # noqa: BLE001 - no active Streamlit session
            pass
    except Exception:  # noqa: BLE001 - streamlit not installed
        pass

    value = os.environ.get(key_name)
    return value if value else None


class NewsAPIProvider(NewsProvider):
    """Provider backed by the NewsAPI.org Everything endpoint."""

    name = "newsapi"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key or resolve_api_key()
        if not self._api_key:
            raise NewsAuthenticationError(
                "NEWSAPI_KEY is required for the NewsAPI provider; set it via "
                "the NEWSAPI_KEY environment variable or Streamlit secrets."
            )
        self._timeout = float(timeout)
        self._session = session or requests.Session()

    def fetch_news(self, symbol: str, *, limit: int = 20) -> list[NewsArticle]:
        params = {
            "q": symbol,
            "language": "en",
            "pageSize": max(1, min(limit, 100)),
            "apiKey": self._api_key,
            "sortBy": "publishedAt",
        }
        try:
            response = self._session.get(_ENDPOINT, params=params, timeout=self._timeout)
        except requests.Timeout as exc:
            raise TimeoutExceededError(f"NewsAPI request timed out for '{symbol}'") from exc
        except requests.RequestException as exc:
            message = str(exc)
            if re.search(r"\b429\b|rate[ _-]*limit|throttl", message, re.IGNORECASE):
                raise RateLimitedError(f"NewsAPI rate limited for '{symbol}'") from exc
            raise NewsProviderError(f"NewsAPI request failed for '{symbol}': {message}") from exc

        if response.status_code == 401:
            raise NewsAuthenticationError("NewsAPI key is invalid (HTTP 401).")
        if response.status_code == 429:
            raise RateLimitedError("NewsAPI rate limited (HTTP 429).")
        if response.status_code >= 400:
            raise NewsProviderError(
                f"NewsAPI returned HTTP {response.status_code} for '{symbol}'"
            )

        payload: Any = response.json()
        if not isinstance(payload, dict):
            return []
        if payload.get("status") != "ok":
            message = str(payload.get("message", "unknown error"))
            if "rate" in message.lower() or "429" in message:
                raise RateLimitedError(f"NewsAPI reported rate limiting: {message}")
            raise NewsProviderError(f"NewsAPI error for '{symbol}': {message}")

        raw_articles = payload.get("articles") or []
        articles: list[NewsArticle] = []
        for item in raw_articles[:limit]:
            article = self._parse(item, symbol=symbol)
            if article is not None:
                articles.append(article)
        return articles

    @staticmethod
    def _parse(item: dict[str, Any], *, symbol: str) -> NewsArticle | None:
        if not isinstance(item, dict):
            return None
        title = normalise_title(item.get("title"))
        if not title:
            return None

        source = ""
        src = item.get("source")
        if isinstance(src, dict):
            source = str(src.get("name", "") or "")
        if not source:
            source = "NewsAPI"

        return NewsArticle(
            symbol=symbol,
            title=title,
            source=source,
            url=str(item.get("url") or ""),
            published_at=parse_timestamp_utc(item.get("publishedAt"))
        or datetime.now(tz=timezone.utc),
            description=normalise_description(item.get("description")),
            provider=NewsAPIProvider.name,
        )
