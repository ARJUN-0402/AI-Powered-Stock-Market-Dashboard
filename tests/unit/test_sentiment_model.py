"""Tests for the finance-domain sentiment models."""

from __future__ import annotations

import pytest

from src.nlp.sentiment_model import (
    FinanceLexiconSentimentModel,
    FinBertSentimentModel,
    SentimentScores,
    create_sentiment_model,
)

# ---------------------------------------------------------------------------
# Lexicon fallback model
# ---------------------------------------------------------------------------


def test_lexicon_classifies_positive_text() -> None:
    model = FinanceLexiconSentimentModel()
    scores = model.classify("Apple reported strong earnings and beat expectations")
    assert scores.label == "Positive"
    assert scores.positive_prob > scores.negative_prob
    assert scores.positive_prob > scores.neutral_prob
    assert abs(scores.positive_prob + scores.neutral_prob + scores.negative_prob - 1.0) < 1e-9


def test_lexicon_classifies_negative_text() -> None:
    model = FinanceLexiconSentimentModel()
    scores = model.classify("Shares tumble on earnings miss and restructuring charges")
    assert scores.label == "Negative"
    assert scores.negative_prob > scores.positive_prob


def test_lexicon_empty_text_is_neutral() -> None:
    model = FinanceLexiconSentimentModel()
    scores = model.classify("")
    assert scores.label == "Neutral"
    assert abs(scores.positive_prob - 1 / 3) < 1e-9


def test_lexicon_is_always_available() -> None:
    assert FinanceLexiconSentimentModel().is_available() is True


# ---------------------------------------------------------------------------
# FinBERT model (transformers pipeline mocked for deterministic tests)
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Mimics ``transformers.pipeline`` output for a single string."""

    def __init__(self, rows: list[dict[str, float]]) -> None:
        self._rows = rows

    def __call__(self, text: str) -> object:
        return [self._rows]


def _finbert_with_pipeline(rows: list[dict[str, float]]) -> FinBertSentimentModel:
    model = FinBertSentimentModel()
    model._handle.pipeline = _FakePipeline(rows)  # type: ignore[union-attr]
    model._handle.error = None
    return model


def test_finbert_normalizes_label_case() -> None:
    rows = [
        {"label": "positive", "score": 0.8},
        {"label": "neutral", "score": 0.15},
        {"label": "negative", "score": 0.05},
    ]
    model = _finbert_with_pipeline(rows)
    scores = model.classify("Some headline")
    assert scores.label == "Positive"
    assert scores.positive_prob == pytest.approx(0.8, abs=1e-6)
    assert model.name == "finbert:finbert-tone"


def test_finbert_aggregates_duplicate_labels() -> None:
    rows = [
        {"label": "Positive", "score": 0.5},
        {"label": "positive", "score": 0.3},
        {"label": "Neutral", "score": 0.2},
    ]
    model = _finbert_with_pipeline(rows)
    scores = model.classify("Some headline")
    assert scores.positive_prob == pytest.approx(0.8, abs=1e-6)
    assert scores.neutral_prob == pytest.approx(0.2, abs=1e-6)


def test_finbert_empty_text_is_neutral() -> None:
    model = _finbert_with_pipeline([{"label": "Positive", "score": 1.0}])
    scores = model.classify("")
    assert scores.label == "Neutral"
    assert abs(scores.positive_prob - 1 / 3) < 1e-9


def test_finbert_falls_back_to_lexicon_when_unloadable() -> None:
    model = FinBertSentimentModel()
    model._handle.error = RuntimeError("simulated load failure")  # type: ignore[union-attr]
    assert model.is_available() is False
    scores = model.classify("Apple beat earnings expectations")
    assert scores.label == "Positive"


def test_create_sentiment_model_prefers_finbert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FinBertSentimentModel, "is_available", lambda self: True)
    monkeypatch.setattr(FinBertSentimentModel, "_load", lambda self: object())
    model = create_sentiment_model()
    assert isinstance(model, FinBertSentimentModel)


def test_create_sentiment_model_falls_back_to_lexicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FinBertSentimentModel, "is_available", lambda self: False)
    model = create_sentiment_model()
    assert isinstance(model, FinanceLexiconSentimentModel)


def test_sentiment_scores_has_expected_fields() -> None:
    scores = SentimentScores(0.6, 0.3, 0.1, "Positive", 0.6)
    assert scores.label == "Positive"
    assert scores.confidence == pytest.approx(0.6)
    assert abs(scores.positive_prob + scores.neutral_prob + scores.negative_prob - 1.0) < 1e-9
