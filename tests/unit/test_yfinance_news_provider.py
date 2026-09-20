"""Tests for the yfinance news provider adapter."""

from __future__ import annotations

import pytest

from src.data.news_service.errors import (
    NewsProviderError,
    RateLimitedError,
    TimeoutExceededError,
)
from src.data.news_service.providers.yfinance_news import (
    YFinanceNewsProvider,
    _looks_like_rate_limit,
    _safe_call,
)
from src.data.news_service.validators import parse_timestamp_utc
from tests.fixtures import make_raw_yfinance_news


class _StubTicker:
    def __init__(self, news: object) -> None:
        self.news = news


class _RaisingTicker:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def news(self) -> object:
        raise self._exc


class _StubYFinanceProvider(YFinanceNewsProvider):
    def __init__(self, news: object | None = None, exc: BaseException | None = None) -> None:
        super().__init__(timeout=0.1)
        self._news = news
        self._exc = exc

    def _ticker(self, symbol: str) -> object:
        if self._exc is not None:
            return _RaisingTicker(self._exc)
        return _StubTicker(self._news)


def test_fetch_news_returns_normalised_articles() -> None:
    provider = _StubYFinanceProvider(news=make_raw_yfinance_news(3))
    articles = provider.fetch_news("AAPL", limit=3)
    assert len(articles) == 3
    assert all(a.symbol == "AAPL" for a in articles)
    assert all(a.provider == "yfinance" for a in articles)
    assert articles[0].source == "Yahoo Finance"
    assert articles[0].url.startswith("https://finance.yahoo.com")


def test_fetch_news_strips_html_description() -> None:
    provider = _StubYFinanceProvider(news=make_raw_yfinance_news(1))
    articles = provider.fetch_news("AAPL", limit=1)
    assert articles[0].description == "Body 1"


def test_parse_extracts_title_description_and_url() -> None:
    item = {
        "content": {
            "title": "Earnings beat",
            "description": "<p>Strong show</p>",
            "summary": "fallback summary",
            "pubDate": "2026-09-01T12:00:00Z",
            "provider": {"displayName": "Yahoo Finance"},
            "canonicalUrl": {"url": "https://example.com/story"},
        }
    }
    article = YFinanceNewsProvider._parse(item, symbol="AAPL")
    assert article is not None
    assert article.title == "Earnings beat"
    assert article.description == "Strong show"
    assert article.url == "https://example.com/story"
    assert article.source == "Yahoo Finance"
    ts = parse_timestamp_utc(item["content"]["pubDate"])
    assert article.published_at == ts


def test_parse_falls_back_to_click_through_url() -> None:
    item = {
        "content": {
            "title": "Headline",
            "pubDate": "2026-09-01T12:00:00Z",
            "provider": {"displayName": "Yahoo Finance"},
            "clickThroughUrl": {"url": "https://yahoo.com/ct"},
        }
    }
    article = YFinanceNewsProvider._parse(item, symbol="AAPL")
    assert article is not None
    assert article.url == "https://yahoo.com/ct"


def test_parse_falls_back_to_summary_description() -> None:
    item = {
        "content": {
            "title": "Headline",
            "summary": "summary text here",
            "pubDate": "2026-09-01T12:00:00Z",
            "provider": {"displayName": "Yahoo Finance"},
            "canonicalUrl": {"url": "https://example.com/s"},
        }
    }
    article = YFinanceNewsProvider._parse(item, symbol="AAPL")
    assert article is not None
    assert article.description == "summary text here"


def test_parse_default_source_when_provider_missing() -> None:
    item = {
        "content": {
            "title": "Headline",
            "pubDate": "2026-09-01T12:00:00Z",
            "canonicalUrl": {"url": "https://example.com/s"},
        }
    }
    article = YFinanceNewsProvider._parse(item, symbol="AAPL")
    assert article is not None
    assert article.source == "Yahoo Finance"


def test_parse_returns_none_for_missing_title() -> None:
    item = {"content": {"pubDate": "2026-09-01T12:00:00Z"}}
    assert YFinanceNewsProvider._parse(item, symbol="AAPL") is None


def test_parse_handles_missing_timestamp() -> None:
    item = {
        "content": {
            "title": "Headline",
            "canonicalUrl": {"url": "https://example.com/s"},
        }
    }
    article = YFinanceNewsProvider._parse(item, symbol="AAPL")
    assert article is not None
    assert article.published_at is not None


def test_fetch_news_translates_timeout() -> None:
    provider = _StubYFinanceProvider(exc=TimeoutError("slow"))
    with pytest.raises(TimeoutExceededError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_rate_limit() -> None:
    provider = _StubYFinanceProvider(exc=RuntimeError("429 too many requests"))
    with pytest.raises(RateLimitedError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_generic_error() -> None:
    provider = _StubYFinanceProvider(exc=RuntimeError("unexpected"))
    with pytest.raises(NewsProviderError):
        provider.fetch_news("AAPL")


def test_safe_call_translates_timeout() -> None:
    def _raise() -> None:
        raise TimeoutError("slow")

    with pytest.raises(TimeoutExceededError):
        _safe_call(_raise, timeout=0.1, label="test")


def test_safe_call_translates_rate_limit() -> None:
    def _raise() -> None:
        raise RuntimeError("429 too many requests")

    with pytest.raises(RateLimitedError):
        _safe_call(_raise, timeout=0.1, label="test")


def test_safe_call_translates_generic_error() -> None:
    def _raise() -> None:
        raise RuntimeError("unexpected")

    with pytest.raises(NewsProviderError):
        _safe_call(_raise, timeout=0.1, label="test")


def test_looks_like_rate_limit_match() -> None:
    assert _looks_like_rate_limit("429 Too Many Requests") is True
    assert _looks_like_rate_limit("rate limit exceeded") is True
    assert _looks_like_rate_limit("throttled") is True
    assert _looks_like_rate_limit("disk full") is False
