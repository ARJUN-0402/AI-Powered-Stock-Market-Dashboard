"""Tests for the centralised news data service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
from src.data.news_service.service import (
    NewsDataService,
    YFinanceNewsProvider,
    get_default_service,
    reset_default_service,
    resolve_provider,
)
from src.nlp.sentiment_model import SentimentModel, SentimentScores
from tests.fixtures import make_article, make_articles

UTC = timezone.utc


class _FakeSentimentModel(SentimentModel):
    name = "test-fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def classify(self, text: str) -> SentimentScores:
        self.calls.append(text)
        lowered = text.lower()
        if any(w in lowered for w in ("loss", "miss", "fall", "drop", "decline")):
            return SentimentScores(0.1, 0.2, 0.7, "Negative", 0.7)
        if any(w in lowered for w in ("beat", "earn", "strong", "profit", "growth")):
            return SentimentScores(0.7, 0.2, 0.1, "Positive", 0.7)
        return SentimentScores(0.2, 0.6, 0.2, "Neutral", 0.6)


class _FakeNewsProvider:
    name = "fake"

    def __init__(
        self,
        articles: list[NewsArticle] | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._articles = list(articles or [])
        self._exc = exc
        self.calls: list[tuple[str, int]] = []

    def fetch_news(self, symbol: str, *, limit: int = 20) -> list[NewsArticle]:
        self.calls.append((symbol, limit))
        if self._exc is not None:
            raise self._exc
        return list(self._articles[:limit])

    def supports_symbol(self, symbol: str) -> bool:
        return True


def _service(
    articles: list[NewsArticle] | None = None,
    model: SentimentModel | None = None,
    exc: BaseException | None = None,
) -> tuple[NewsDataService, _FakeNewsProvider, _FakeSentimentModel]:
    provider = _FakeNewsProvider(articles=articles, exc=exc)
    sentiment = model or _FakeSentimentModel()
    svc = NewsDataService(provider=provider, sentiment_model=sentiment)
    return svc, provider, sentiment


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_symbol_raises() -> None:
    svc, _, _ = _service()
    with pytest.raises(InvalidSymbolError):
        svc.fetch_articles("BAD!")


def test_empty_symbol_raises() -> None:
    svc, _, _ = _service()
    with pytest.raises(InvalidSymbolError):
        svc.get_news_with_sentiment("")


# ---------------------------------------------------------------------------
# Happy path / normalisation
# ---------------------------------------------------------------------------


def test_fetch_articles_returns_sorted_recent_articles() -> None:
    articles = make_articles(5)
    svc, provider, _ = _service(articles=articles)
    result = svc.fetch_articles("AAPL", limit=5)
    assert len(result) == 5
    assert provider.calls == [("AAPL", 5)]
    # sorted descending by published_at
    assert result[0].published_at >= result[-1].published_at


def test_get_news_with_sentiment_attaches_model() -> None:
    svc, _, sentiment = _service(articles=make_articles(3))
    scored = svc.get_news_with_sentiment("AAPL")
    assert len(scored) == 3
    assert all(isinstance(s, ArticleSentiment) for s in scored)
    assert all(s.model == "test-fake" for s in scored)
    assert len(sentiment.calls) == 3


def test_get_news_with_sentiment_classifies_by_content() -> None:
    articles = [
        make_article(title="Apple beats earnings expectations", url="https://a.com/1"),
        make_article(title="Shares slide after miss", url="https://a.com/2"),
        make_article(
            title="Quarterly update",
            url="https://a.com/3",
            description=None,
        ),
    ]
    svc, _, _ = _service(articles=articles)
    scored = svc.get_news_with_sentiment("AAPL")
    labels = [s.label for s in scored]
    assert labels == ["Positive", "Negative", "Neutral"]


def test_classify_uses_title_and_description() -> None:
    article = make_article(
        title="Report", description="Loss reported here", url="https://a.com/1"
    )
    svc, _, sentiment = _service(articles=[article])
    svc.get_news_with_sentiment("AAPL")
    assert "Loss reported here" in sentiment.calls[0]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_get_aggregate_sentiment_recent() -> None:
    svc, _, _ = _service(articles=make_articles(5))
    agg = svc.get_aggregate_sentiment("AAPL", window="recent")
    assert isinstance(agg, AggregateSentiment)
    assert agg.window == "recent"
    assert agg.article_count == 5
    assert agg.provider == "fake"
    assert agg.model == "test-fake"
    assert agg.positive_prob + agg.neutral_prob + agg.negative_prob == pytest.approx(1.0)


def test_get_aggregate_sentiment_daily_window() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    articles = make_articles(3, base=now)
    # one old article outside the 24h daily window
    old = make_article(
        title="Old news", url="https://old.com", published_at=now - timedelta(days=3)
    )
    svc, _, _ = _service(articles=[*articles, old])
    agg = svc.get_aggregate_sentiment("AAPL", window="daily")
    assert agg.window == "daily"
    assert agg.article_count == 3
    assert agg.window_end == now


def test_get_aggregate_sentiment_weekly_window() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    articles = make_articles(2, base=now)
    old = make_article(
        title="Old news", url="https://old.com", published_at=now - timedelta(days=20)
    )
    svc, _, _ = _service(articles=[*articles, old])
    agg = svc.get_aggregate_sentiment("AAPL", window="weekly")
    assert agg.window == "weekly"
    assert agg.article_count == 2


def test_get_aggregate_sentiment_unknown_window_raises() -> None:
    svc, _, _ = _service(articles=make_articles(2))
    with pytest.raises(NewsDataError):
        svc.get_aggregate_sentiment("AAPL", window="monthly")


def test_get_aggregate_sentiment_empty_symbol() -> None:
    svc, _, _ = _service(articles=[])
    agg = svc.get_aggregate_sentiment("AAPL", window="recent")
    assert agg.article_count == 0
    assert agg.label == "Neutral"


def test_empty_article_set_classifies_to_neutral() -> None:
    scores = _FakeSentimentModel().classify("")
    assert scores.label == "Neutral"


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_get_news_trend_returns_daily_buckets() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    articles = make_articles(
        3, base=now, offset_hours=0
    )
    # add an article from a different day
    other = make_article(
        title="beat", url="https://a.com/x", published_at=now - timedelta(days=2)
    )
    svc, _, _ = _service(articles=[*articles, other])
    trend = svc.get_news_trend("AAPL", days=7)
    assert len(trend) >= 1
    assert {"date", "label", "sentiment", "count", "model"} <= trend[0].keys()
    assert trend[-1]["count"] >= 1


def test_get_news_trend_empty() -> None:
    svc, _, _ = _service(articles=[])
    assert svc.get_news_trend("AAPL") == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_caching_avoids_repeat_provider_call() -> None:
    articles = make_articles(3)
    svc, provider, _ = _service(articles=articles)
    first = svc.get_news_with_sentiment("AAPL")
    second = svc.get_news_with_sentiment("AAPL")
    assert first == second
    assert len(provider.calls) == 1


def test_caching_bypassed_with_use_cache_false() -> None:
    svc, provider, _ = _service(articles=make_articles(3))
    svc.get_news_with_sentiment("AAPL", use_cache=False)
    svc.get_news_with_sentiment("AAPL", use_cache=False)
    assert len(provider.calls) == 2


def test_refresh_invalidates_symbol_cache() -> None:
    svc, provider, _ = _service(articles=make_articles(3))
    svc.get_news_with_sentiment("AAPL")
    svc.get_news_with_sentiment("MSFT")
    assert len(provider.calls) == 2
    svc.refresh("AAPL")
    svc.get_news_with_sentiment("AAPL")
    svc.get_news_with_sentiment("MSFT")
    assert len(provider.calls) == 3


def test_refresh_clears_cache() -> None:
    svc, provider, _ = _service(articles=make_articles(3))
    svc.get_news_with_sentiment("AAPL")
    svc.refresh()
    svc.get_news_with_sentiment("AAPL")
    assert len(provider.calls) == 2


def test_caching_distinguishes_symbols() -> None:
    svc, provider, _ = _service(articles=make_articles(3))
    svc.get_news_with_sentiment("AAPL")
    svc.get_news_with_sentiment("MSFT")
    assert [c[0] for c in provider.calls] == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Deduplication & recency at the service layer
# ---------------------------------------------------------------------------


def test_service_deduplicates_similar_headlines() -> None:
    a = make_article(title="Apple beats earnings expectations badly", url="https://a.com")
    b = make_article(title="Apple beats earnings expectations badly", url="https://b.com")
    svc, provider, _ = _service(articles=[a, b])
    result = svc.fetch_articles("AAPL")
    assert len(result) == 1
    assert len(provider.calls) == 1


def test_service_filters_old_articles() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    recent = make_article(title="Recent", url="https://a.com", published_at=now)
    stale = make_article(
        title="Stale", url="https://b.com", published_at=now - timedelta(days=10)
    )
    svc, _, _ = _service(articles=[recent, stale])
    result = svc.fetch_articles("AAPL")
    titles = [a.title for a in result]
    assert "Recent" in titles
    assert "Stale" not in titles


def test_service_sorts_descending() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    first = make_article(title="Newer", url="https://a.com", published_at=now)
    second = make_article(title="Older", url="https://b.com", published_at=now - timedelta(hours=2))
    svc, _, _ = _service(articles=[second, first])
    result = svc.fetch_articles("AAPL")
    assert result[0].title == "Newer"
    assert result[1].title == "Older"


# ---------------------------------------------------------------------------
# Error handling / graceful degradation
# ---------------------------------------------------------------------------


def test_provider_exception_propagates() -> None:
    svc, _, _ = _service(exc=NewsProviderError("boom"))
    with pytest.raises(NewsProviderError):
        svc.fetch_articles("AAPL")


def test_unexpected_exception_is_wrapped() -> None:
    class _FakeProvider(_FakeNewsProvider):
        def fetch_news(self, symbol: str, *, limit: int = 20) -> list[NewsArticle]:
            raise RuntimeError("nope")

    svc = NewsDataService(provider=_FakeProvider(), sentiment_model=_FakeSentimentModel())
    with pytest.raises(NewsProviderError):
        svc.fetch_articles("AAPL")


def test_rate_limit_propagates() -> None:
    svc, _, _ = _service(exc=RateLimitedError("throttled"))
    with pytest.raises(RateLimitedError):
        svc.get_news_with_sentiment("AAPL")


def test_timeout_propagates() -> None:
    svc, _, _ = _service(exc=TimeoutExceededError("slow"))
    with pytest.raises(TimeoutExceededError):
        svc.get_news_with_sentiment("AAPL")


def test_authentication_error_propagates() -> None:
    svc, _, _ = _service(exc=NewsAuthenticationError("no key"))
    with pytest.raises(NewsAuthenticationError):
        svc.get_news_with_sentiment("AAPL")


def test_provider_status_attributes() -> None:
    svc, _, _ = _service(articles=make_articles(3))
    svc.get_news_with_sentiment("AAPL")
    status = svc.provider_status(3)
    assert isinstance(status, ProviderStatus)
    assert status.name == "fake"
    assert status.model == "test-fake"
    assert status.is_real is True
    assert status.is_fabricated is False


def test_configure_updates_limit() -> None:
    svc, _, _ = _service(articles=make_articles(5))
    svc.configure(limit=2)
    result = svc.fetch_articles("AAPL")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def test_resolve_provider_yfinance() -> None:
    provider = resolve_provider("yfinance")
    assert isinstance(provider, YFinanceNewsProvider)
    assert provider.name == "yfinance"


def test_resolve_provider_unknown_raises() -> None:
    with pytest.raises(NewsDataError):
        resolve_provider("does-not-exist")


def test_get_default_service_is_lazy() -> None:
    reset_default_service()
    svc = get_default_service()
    assert isinstance(svc, NewsDataService)
    reset_default_service()
