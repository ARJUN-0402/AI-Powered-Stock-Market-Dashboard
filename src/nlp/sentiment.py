"""Sentiment analysis over news headlines.

This module is the backward-compatible face of sentiment analysis. It now
delegates to the finance-domain models in :mod:`src.nlp.sentiment_model`
(FinBERT, with a Loughran-McDonald lexicon fallback) rather than the legacy
TextBlob polarity heuristic, so classifications are finance-aware.

The public functions (``analyze_text``, ``analyze_headlines``,
``build_news_items``, ``aggregate_sentiment``) and the
:class:`SentimentResult` dataclass are preserved for existing callers
(including the Streamlit dashboard). ``SentimentResult`` has been enriched
with per-class probabilities and a confidence score while keeping the
``label`` and ``polarity`` fields that older code relies on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.nlp.sentiment_model import (
    SentimentScores,
    get_default_sentiment_model,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SentimentResult:
    """Container for a sentiment classification.

    ``polarity`` is retained for backward compatibility and is derived as
    ``positive_prob - negative_prob`` (range ``[-1, 1]``), mirroring the
    sign semantics of the previous TextBlob implementation. Callers that
    need the full distribution should read the ``*_prob`` fields.
    """

    label: str
    polarity: float
    positive_prob: float = 0.0
    neutral_prob: float = 0.0
    negative_prob: float = 0.0
    confidence: float = 0.0

    @classmethod
    def from_scores(cls, scores: SentimentScores) -> SentimentResult:
        """Build a result from a :class:`SentimentScores` payload."""

        polarity = scores.positive_prob - scores.negative_prob
        return cls(
            label=scores.label,
            polarity=polarity,
            positive_prob=scores.positive_prob,
            neutral_prob=scores.neutral_prob,
            negative_prob=scores.negative_prob,
            confidence=scores.confidence,
        )


def analyze_text(text: str) -> SentimentResult:
    """Compute sentiment for a single piece of text."""

    if not text:
        return SentimentResult(label="Neutral", polarity=0.0)
    try:
        model = get_default_sentiment_model()
        return SentimentResult.from_scores(model.classify(str(text)))
    except Exception as exc:  # noqa: BLE001 - model failure must not crash callers
        logger.debug("Sentiment analysis failed for text: %s", exc)
        return SentimentResult(label="Neutral", polarity=0.0)


def analyze_headlines(headlines: Iterable[str]) -> list[SentimentResult]:
    """Compute sentiment for a sequence of headlines."""

    return [analyze_text(headline) for headline in headlines]


def build_news_items(headlines: Iterable[str]) -> list[dict]:
    """Combine headlines with sentiment results into dictionaries.

    The returned dictionaries retain the historical ``title``,
    ``sentiment`` and ``polarity`` keys for backward compatibility while
    exposing the richer probability distribution.
    """

    results: list[dict] = []
    for headline in headlines:
        result = analyze_text(headline)
        results.append(
            {
                "title": headline,
                "sentiment": result.label,
                "polarity": result.polarity,
                "positive_prob": result.positive_prob,
                "neutral_prob": result.neutral_prob,
                "negative_prob": result.negative_prob,
                "confidence": result.confidence,
            }
        )
    return results


def aggregate_sentiment(results: Iterable[SentimentResult] | None) -> float:
    """Return the mean polarity across ``results`` (0.0 if none)."""

    if not results:
        return 0.0
    values = [r.polarity for r in results if r is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))
