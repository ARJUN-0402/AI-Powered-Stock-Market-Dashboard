"""Tests for the NewsAPI.org provider adapter."""

from __future__ import annotations

import pytest
import requests

from src.data.news_service.errors import (
    NewsAuthenticationError,
    NewsProviderError,
    RateLimitedError,
    TimeoutExceededError,
)
from src.data.news_service.providers.newsapi_provider import (
    NewsAPIProvider,
    resolve_api_key,
)
from tests.fixtures import make_raw_newsapi_articles


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data: object = None) -> None:
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = "error body"

    def json(self) -> object:
        return self._json


class _FakeSession:
    def __init__(
        self,
        response: _FakeResponse | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._response = response
        self._exc = exc
        self.calls: list[tuple[str, dict]] = []

    def get(
        self,
        url: str,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls.append((url, params or {}))
        if self._exc is not None:
            raise self._exc
        return self._response  # type: ignore[return-value]


def _provider_with(
    response: _FakeResponse | None = None,
    exc: BaseException | None = None,
) -> tuple[NewsAPIProvider, _FakeSession]:
    session = _FakeSession(response=response, exc=exc)
    provider = NewsAPIProvider(api_key="test-key", session=session)  # type: ignore[arg-type]
    return provider, session


def test_construction_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data.news_service.providers.newsapi_provider.resolve_api_key",
        lambda name="NEWSAPI_KEY": None,
    )
    try:
        NewsAPIProvider(api_key=None)
    except NewsAuthenticationError as exc:
        assert "NEWSAPI_KEY" in str(exc)
    else:
        pytest.fail("NewsAuthenticationError was expected")


def test_resolve_api_key_checks_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    assert resolve_api_key() is None
    monkeypatch.setenv("NEWSAPI_KEY", "secret")
    assert resolve_api_key() == "secret"


def test_fetch_news_parses_articles() -> None:
    provider, session = _provider_with(
        response=_FakeResponse(200, {"status": "ok", "articles": make_raw_newsapi_articles(3)})
    )
    articles = provider.fetch_news("AAPL", limit=3)
    assert len(articles) == 3
    assert articles[0].title == "Market headline 1"
    assert articles[0].source == "Reuters"
    assert articles[0].url == "https://reuters.com/article/0"
    assert articles[0].provider == "newsapi"
    assert session.calls[0][1]["q"] == "AAPL"


def test_fetch_news_skips_items_missing_title() -> None:
    payload = {"status": "ok", "articles": [{"title": None, "url": "x"}]}
    provider, _ = _provider_with(response=_FakeResponse(200, payload))
    assert provider.fetch_news("AAPL", limit=5) == []


def test_fetch_news_handles_empty_response() -> None:
    provider, _ = _provider_with(response=_FakeResponse(200, {"status": "ok", "articles": []}))
    assert provider.fetch_news("AAPL") == []


def test_fetch_news_translates_auth_error() -> None:
    provider, _ = _provider_with(response=_FakeResponse(401, {"message": "invalid key"}))
    with pytest.raises(NewsAuthenticationError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_rate_limit() -> None:
    provider, _ = _provider_with(response=_FakeResponse(429, {"message": "slow down"}))
    with pytest.raises(RateLimitedError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_server_error() -> None:
    provider, _ = _provider_with(response=_FakeResponse(500, {"message": "boom"}))
    with pytest.raises(NewsProviderError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_api_message_error() -> None:
    payload = {"status": "error", "code": "rate_limit", "message": "rate limit exceeded"}
    provider, _ = _provider_with(response=_FakeResponse(200, payload))
    with pytest.raises(RateLimitedError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_timeout() -> None:
    provider, _ = _provider_with(exc=requests.Timeout("timed out"))
    with pytest.raises(TimeoutExceededError):
        provider.fetch_news("AAPL")


def test_fetch_news_translates_connection_error() -> None:
    provider, _ = _provider_with(exc=requests.ConnectionError("no network"))
    with pytest.raises(NewsProviderError):
        provider.fetch_news("AAPL")
