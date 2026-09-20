"""Tests for the news service validation/normalisation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.news_service.validators import (
    deduplicate_articles,
    filter_recent_articles,
    normalise_description,
    normalise_title,
    parse_timestamp_utc,
    validate_article,
)
from tests.fixtures import make_article


def test_parse_timestamp_iso_with_z() -> None:
    ts = parse_timestamp_utc("2026-09-01T12:00:00Z")
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts == datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_timestamp_iso_with_offset() -> None:
    ts = parse_timestamp_utc("2026-09-01T14:00:00+02:00")
    assert ts is not None
    assert ts.utcoffset() == timezone.utc.utcoffset(None)
    assert ts.hour == 12


def test_parse_timestamp_rfc2822() -> None:
    ts = parse_timestamp_utc("Mon, 31 Aug 2026 10:00:00 GMT")
    assert ts is not None
    assert ts.year == 2026 and ts.month == 8 and ts.day == 31
    assert ts.tzinfo is not None


def test_parse_timestamp_epoch_seconds() -> None:
    ts = parse_timestamp_utc(1_750_000_000)
    assert ts is not None
    assert ts.tzinfo is not None


def test_parse_timestamp_naive_datetime_promoted_to_utc() -> None:
    naive = datetime(2026, 9, 1, 12, 0, 0)
    ts = parse_timestamp_utc(naive)
    assert ts is not None
    assert ts.tzinfo is timezone.utc


def test_parse_timestamp_invalid_returns_none() -> None:
    assert parse_timestamp_utc("not a date") is None
    assert parse_timestamp_utc(None) is None
    assert parse_timestamp_utc("") is None


def test_normalise_title_strips_html_and_whitespace() -> None:
    title = normalise_title("  <p>Hello   <b>World</b></p>  ")
    assert title == "Hello World"


def test_normalise_title_truncates_long_titles() -> None:
    title = normalise_title("a" * 500)
    assert len(title) <= 300


def test_normalise_description_none_and_empty() -> None:
    assert normalise_description(None) is None
    assert normalise_description("   <b></b>  ") is None
    assert normalise_description("<p>Real body</p>") == "Real body"


def test_validate_article_accepts_valid_article() -> None:
    assert validate_article(make_article()) is True


def test_validate_article_rejects_missing_title() -> None:
    article = make_article(title="   ")
    assert validate_article(article) is False


def test_validate_article_rejects_missing_url() -> None:
    article = make_article(url="")
    assert validate_article(article) is False


def test_validate_article_rejects_missing_source() -> None:
    article = make_article(source="")
    assert validate_article(article) is False


def test_validate_article_rejects_naive_timestamp() -> None:
    naive = datetime(2026, 9, 1, 12, 0, 0)
    article = make_article(published_at=naive)
    assert validate_article(article) is False


def test_validate_article_rejects_non_article() -> None:
    assert validate_article({"title": "x"}) is False


def test_deduplicate_removes_exact_url_duplicates() -> None:
    a = make_article(url="https://example.com/1", title="First")
    b = make_article(url="https://example.com/1", title="Duplicate url")
    result = deduplicate_articles([a, b], similarity_threshold=0.5)
    assert len(result) == 1
    assert result[0].title == "First"


def test_deduplicate_removes_near_duplicate_titles() -> None:
    a = make_article(title="Apple reports earnings beat expectations", url="https://a.com")
    b = make_article(title="Apple reports earnings beat expectations.", url="https://b.com")
    result = deduplicate_articles([a, b], similarity_threshold=0.9)
    assert len(result) == 1


def test_deduplicate_keeps_distinct_titles() -> None:
    a = make_article(title="Apple beats earnings", url="https://a.com")
    b = make_article(title="Tesla misses deliveries", url="https://b.com")
    result = deduplicate_articles([a, b], similarity_threshold=0.9)
    assert len(result) == 2


def test_deduplicate_preserves_order() -> None:
    a = make_article(title="First story", url="https://a.com")
    b = make_article(title="Second story", url="https://b.com")
    result = deduplicate_articles([a, b], similarity_threshold=0.9)
    assert result[0].title == "First story"


def test_filter_recent_drops_old_articles() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = make_article(published_at=now)
    old = make_article(published_at=now - timedelta(hours=25))
    result = filter_recent_articles([recent, old], hours=24)
    assert len(result) == 1
    assert result[0].title == recent.title


def test_filter_recent_keeps_all_within_window() -> None:
    base = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    articles = [
        make_article(published_at=base),
        make_article(published_at=base - timedelta(hours=1)),
    ]
    result = filter_recent_articles(articles, hours=72)
    assert len(result) == 2


def test_filter_recent_empty_input() -> None:
    assert filter_recent_articles([], hours=72) == []
