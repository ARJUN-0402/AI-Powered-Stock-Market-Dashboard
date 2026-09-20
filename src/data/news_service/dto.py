"""Data Transfer Objects for the news data service.

These value objects decouple the dashboard from provider-specific shapes.
Providers translate their native response into :class:`NewsArticle`; the
service attaches sentiment to produce :class:`ArticleSentiment`; and the
aggregation layer produces :class:`AggregateSentiment`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class NewsArticle:
    """A single normalised news item.

    ``published_at`` is always timezone-aware (UTC) or, when the provider
    could not supply one, the fetch time so that downstream recency
    filtering never raises on naive timestamps. Values are never fabricated:
    a genuinely unknown field is ``None`` or an empty string, not a guess.
    """

    symbol: str
    title: str
    source: str
    url: str
    published_at: datetime = field(default_factory=_utcnow)
    description: str | None = None
    provider: str = ""

    @classmethod
    def unavailable(cls, symbol: str, provider: str = "") -> NewsArticle:
        return cls(
            symbol=symbol,
            title="",
            source="",
            url="",
            published_at=_utcnow(),
            description=None,
            provider=provider,
        )


@dataclass(frozen=True)
class ArticleSentiment:
    """A :class:`NewsArticle` paired with its sentiment classification."""

    article: NewsArticle
    positive_prob: float
    neutral_prob: float
    negative_prob: float
    label: str
    confidence: float
    model: str


@dataclass(frozen=True)
class AggregateSentiment:
    """Aggregated sentiment over a time window.

    The probabilities are the mean of the underlying article probabilities.
    ``confidence`` is the probability mass of the winning label, not a
    claim about future price movement.
    """

    symbol: str
    window: str
    window_start: datetime
    window_end: datetime
    positive_prob: float
    neutral_prob: float
    negative_prob: float
    label: str
    confidence: float
    article_count: int
    provider: str
    model: str


@dataclass(frozen=True)
class ProviderStatus:
    """Metadata the service exposes so the UI can attribute sentiment."""

    name: str
    model: str
    is_real: bool
    article_count: int = 0

    @property
    def is_fabricated(self) -> bool:
        return not self.is_real
