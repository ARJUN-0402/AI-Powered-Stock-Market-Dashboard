"""Fixtures used by unit and integration tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.data.news_service.dto import ArticleSentiment, NewsArticle


def make_price_frame(rows: int = 80, start: float = 100.0) -> pd.DataFrame:
    """Return a deterministic OHLCV frame with a gentle uptrend."""

    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    closes = start + np.linspace(0, 10, rows)
    opens = closes + np.random.default_rng(42).normal(0, 0.5, rows)
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    volumes = np.random.default_rng(7).integers(100_000, 500_000, rows)
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


def make_constant_frame(rows: int = 30, price: float = 100.0) -> pd.DataFrame:
    """Return a flat OHLCV frame useful for boundary testing."""

    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": price,
            "High": price,
            "Low": price,
            "Close": price,
            "Volume": 100_000,
        },
        index=dates,
    )


def ohlcv_tuple() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a tuple with a price frame and a constant frame."""

    return make_price_frame(), make_constant_frame()


# ---------------------------------------------------------------------------
# News fixtures
# ---------------------------------------------------------------------------


def make_article(
    symbol: str = "AAPL",
    title: str = "Apple reports strong quarterly earnings",
    source: str = "Yahoo Finance",
    url: str = "https://example.com/articles/1",
    published_at: datetime | None = None,
    description: str | None = "Apple reported strong quarterly earnings.",
    provider: str = "yfinance",
) -> NewsArticle:
    """Return a single normalised :class:`NewsArticle`."""

    if published_at is None:
        published_at = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    return NewsArticle(
        symbol=symbol,
        title=title,
        source=source,
        url=url,
        published_at=published_at,
        description=description,
        provider=provider,
    )


# Distinct headline templates used by :func:`make_articles`. Titles must be
# sufficiently different that the service's headline-similarity deduplication
# (default threshold 0.9) keeps every one of them — otherwise multi-article
# tests silently collapse to a single article.
_DISTINCT_TITLES = (
    "Apple reports record quarterly revenue",
    "New product launch expands market reach",
    "Analysts revise outlook after earnings call",
    "Supply chain improvements lower production costs",
    "Regulatory filing details executive compensation",
    "Partnership agreement signed with major distributor",
    "Innovation pipeline adds three candidate therapies",
    "Share repurchase programme authorised by board",
    "International expansion opens second manufacturing hub",
    "Customer satisfaction scores reach all time high",
    "Research division publishes breakthrough study",
    "Dividend policy reviewed for current fiscal year",
    "Cloud infrastructure upgrade completes on schedule",
    "Executive team announces strategic reorganisation",
    "Sustainability targets met ahead of deadline",
)


def make_articles(
    n: int = 5,
    symbol: str = "AAPL",
    *,
    base: datetime | None = None,
    offset_hours: int = 0,
) -> list[NewsArticle]:
    """Return ``n`` deterministic, chronologically ordered articles.

    Titles are drawn from :data:`_DISTINCT_TITLES` so that the service's
    headline-similarity deduplication (default threshold ``0.9``) keeps every
    article instead of collapsing near-identical strings.
    """

    anchor = base or datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    articles: list[NewsArticle] = []
    for i in range(n):
        ts = anchor - timedelta(hours=offset_hours + i)
        title = _DISTINCT_TITLES[i % len(_DISTINCT_TITLES)]
        articles.append(
            make_article(
                symbol=symbol,
                title=title,
                url=f"https://example.com/articles/{i}",
                published_at=ts,
            )
        )
    return articles


def make_article_sentiment(
    article: NewsArticle | None = None,
    *,
    label: str = "Positive",
    positive_prob: float = 0.9,
    neutral_prob: float = 0.05,
    negative_prob: float = 0.05,
    confidence: float = 0.9,
    model: str = "test",
) -> ArticleSentiment:
    """Return an :class:`ArticleSentiment` wrapping a fixture article."""

    if article is None:
        article = make_article()
    return ArticleSentiment(
        article=article,
        positive_prob=positive_prob,
        neutral_prob=neutral_prob,
        negative_prob=negative_prob,
        label=label,
        confidence=confidence,
        model=model,
    )


def make_raw_yfinance_news(n: int = 3) -> list[dict]:
    """Return raw yfinance-style news payloads (as returned by the API)."""

    items: list[dict] = []
    for i in range(n):
        items.append(
            {
                "content": {
                    "contentType": "STORY",
                    "title": f"Apple headline {i + 1}",
                    "description": f"<p>Body {i + 1}</p>",
                    "summary": f"Summary {i + 1}",
                    "pubDate": "2026-09-01T12:00:00Z",
                    "provider": {"displayName": "Yahoo Finance"},
                    "canonicalUrl": {"url": f"https://finance.yahoo.com/news/{i}"},
                    "clickThroughUrl": {"url": f"https://finance.yahoo.com/news/{i}"},
                }
            }
        )
    return items


def make_raw_newsapi_articles(n: int = 3) -> list[dict]:
    """Return raw NewsAPI-style article payloads."""

    articles: list[dict] = []
    for i in range(n):
        articles.append(
            {
                "source": {"name": "Reuters"},
                "title": f"Market headline {i + 1}",
                "description": f"Details {i + 1}",
                "url": f"https://reuters.com/article/{i}",
                "publishedAt": "2026-09-01T12:00:00Z",
            }
        )
    return articles
