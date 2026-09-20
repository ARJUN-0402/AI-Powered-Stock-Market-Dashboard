"""Sentiment aggregation over time windows.

The aggregation layer turns a collection of per-article :class:`ArticleSentiment`
objects into :class:`AggregateSentiment` snapshots for the *daily*, *weekly*
and *recent-news* windows. It never fabricates data: an empty window yields a
neutral aggregate with ``article_count == 0`` so the UI can render an honest
"no data" state.

Aggregation always works on article *probabilities*, not on the discrete
labels, which avoids the bias that hard label voting introduces when article
counts are small.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from statistics import fmean

from src.data.news_service.dto import AggregateSentiment, ArticleSentiment
from src.utils.logging import get_logger

logger = get_logger(__name__)

_WINDOW_LABELS = ("daily", "weekly", "recent")

_ZERO = AggregateSentiment(
    symbol="",
    window="recent",
    window_start=datetime.now(tz=timezone.utc),
    window_end=datetime.now(tz=timezone.utc),
    positive_prob=1 / 3,
    neutral_prob=1 / 3,
    negative_prob=1 / 3,
    label="Neutral",
    confidence=1 / 3,
    article_count=0,
    provider="",
    model="",
)


def _argmax_label(positive: float, neutral: float, negative: float) -> tuple[str, float]:
    scores = {"Positive": positive, "Neutral": neutral, "Negative": negative}
    label = max(scores, key=scores.get)  # type: ignore[arg-type]
    return label, scores[label]


def _mean_scores(articles: Sequence[ArticleSentiment]) -> tuple[float, float, float, float, int]:
    n = len(articles)
    if n == 0:
        third = 1 / 3
        return third, third, third, third, 0
    positive = fmean(a.positive_prob for a in articles)
    neutral = fmean(a.neutral_prob for a in articles)
    negative = fmean(a.negative_prob for a in articles)
    label, confidence = _argmax_label(positive, neutral, negative)
    return positive, neutral, negative, confidence, n


def _aggregate(
    articles: Sequence[ArticleSentiment],
    *,
    symbol: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    provider: str,
    model: str,
) -> AggregateSentiment:
    positive, neutral, negative, confidence, n = _mean_scores(articles)
    label, _ = _argmax_label(positive, neutral, negative)
    return AggregateSentiment(
        symbol=symbol,
        window=window,
        window_start=window_start,
        window_end=window_end,
        positive_prob=positive,
        neutral_prob=neutral,
        negative_prob=negative,
        label=label,
        confidence=confidence,
        article_count=n,
        provider=provider,
        model=model,
    )


def empty_aggregate(
    *,
    symbol: str = "",
    window: str = "recent",
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    provider: str = "",
    model: str = "",
) -> AggregateSentiment:
    """Return an honest neutral aggregate used when no articles exist."""

    now = datetime.now(tz=timezone.utc)
    return AggregateSentiment(
        symbol=symbol,
        window=window,
        window_start=window_start or now,
        window_end=window_end or now,
        positive_prob=1 / 3,
        neutral_prob=1 / 3,
        negative_prob=1 / 3,
        label="Neutral",
        confidence=1 / 3,
        article_count=0,
        provider=provider,
        model=model,
    )


def anchor_time(articles: Sequence[ArticleSentiment]) -> datetime:
    """Return the newest article timestamp (UTC), else ``now``."""

    if not articles:
        return datetime.now(tz=timezone.utc)
    return max(a.article.published_at for a in articles)


def filter_window(
    articles: Sequence[ArticleSentiment],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[ArticleSentiment]:
    """Keep articles whose publication timestamp falls in ``[start, end]``."""

    return [
        a
        for a in articles
        if window_start <= a.article.published_at <= window_end
    ]


def aggregate_daily(
    articles: Sequence[ArticleSentiment],
    *,
    symbol: str,
    provider: str = "",
    model: str = "",
) -> AggregateSentiment:
    if not articles:
        now = datetime.now(tz=timezone.utc)
        return empty_aggregate(symbol=symbol, window="daily", window_start=now, window_end=now)
    end = anchor_time(articles)
    start = end - timedelta(days=1)
    in_window = filter_window(articles, window_start=start, window_end=end)
    return _aggregate(
        in_window,
        symbol=symbol,
        window="daily",
        window_start=start,
        window_end=end,
        provider=provider,
        model=model,
    )


def aggregate_weekly(
    articles: Sequence[ArticleSentiment],
    *,
    symbol: str,
    provider: str = "",
    model: str = "",
) -> AggregateSentiment:
    if not articles:
        now = datetime.now(tz=timezone.utc)
        return empty_aggregate(symbol=symbol, window="weekly", window_start=now, window_end=now)
    end = anchor_time(articles)
    start = end - timedelta(days=7)
    in_window = filter_window(articles, window_start=start, window_end=end)
    return _aggregate(
        in_window,
        symbol=symbol,
        window="weekly",
        window_start=start,
        window_end=end,
        provider=provider,
        model=model,
    )


def aggregate_recent(
    articles: Sequence[ArticleSentiment],
    *,
    symbol: str,
    hours: int,
    provider: str = "",
    model: str = "",
) -> AggregateSentiment:
    if not articles:
        now = datetime.now(tz=timezone.utc)
        return empty_aggregate(symbol=symbol, window="recent", window_start=now, window_end=now)
    end = anchor_time(articles)
    start = end - timedelta(hours=hours)
    in_window = filter_window(articles, window_start=start, window_end=end)
    return _aggregate(
        in_window,
        symbol=symbol,
        window="recent",
        window_start=start,
        window_end=end,
        provider=provider,
        model=model,
    )


def aggregate_market(
    articles: Sequence[ArticleSentiment],
    *,
    symbols: Sequence[str],
    hours: int = 72,
    provider: str = "",
    model: str = "",
) -> AggregateSentiment:
    """Aggregate sentiment across a basket of tickers (market-wide view)."""

    if not articles:
        now = datetime.now(tz=timezone.utc)
        return empty_aggregate(
            symbol="MARKET", window="recent", window_start=now, window_end=now
        )
    end = anchor_time(articles)
    start = end - timedelta(hours=hours)
    in_window = filter_window(articles, window_start=start, window_end=end)
    return _aggregate(
        in_window,
        symbol="MARKET",
        window="recent",
        window_start=start,
        window_end=end,
        provider=provider,
        model=model,
    )
