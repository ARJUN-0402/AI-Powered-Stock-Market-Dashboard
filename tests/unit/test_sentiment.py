"""Tests for sentiment analysis helpers."""

from __future__ import annotations

from src.nlp.sentiment import (
    aggregate_sentiment,
    analyze_headlines,
    analyze_text,
    build_news_items,
)


def test_analyze_text_returns_neutral_for_empty() -> None:
    result = analyze_text("")
    assert result.label == "Neutral"
    assert result.polarity == 0.0


def test_analyze_headlines_produces_results() -> None:
    results = analyze_headlines(["Great quarter", "Bad news", ""])
    assert len(results) == 3
    assert all(result.label in {"Positive", "Negative", "Neutral"} for result in results)


def test_build_news_items_includes_metadata() -> None:
    items = build_news_items(["Apple launches a new product", "Concerns over supply chain"])
    assert len(items) == 2
    for item in items:
        assert {"title", "sentiment", "polarity"} <= item.keys()


def test_aggregate_sentiment_handles_empty_input() -> None:
    assert aggregate_sentiment([]) == 0.0
    assert aggregate_sentiment(None) == 0.0  # type: ignore[arg-type]


def test_aggregate_sentiment_averages_polarity() -> None:
    results = [analyze_text("Great"), analyze_text("Terrible")]
    avg = aggregate_sentiment(results)
    expected = sum(r.polarity for r in results) / len(results)
    assert abs(avg - expected) < 1e-9
