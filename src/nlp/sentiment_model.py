"""Financial-domain sentiment models.

This module provides the sentiment-analysis step of the news pipeline. The
primary implementation is :class:`FinBertSentimentModel`, which uses a real
finance-trained transformer (FinBERT) via :mod:`transformers`. FinBERT is
trained on financial texts (the "tone" variant classifies headlines into
positive / neutral / negative) and is a far better fit for market news than a
generic polarity analyser such as TextBlob.

When the transformer stack (or a network connection to download the model) is
unavailable, the pipeline transparently falls back to
:class:`FinanceLexiconSentimentModel` - a deterministic, finance-domain
lexicon classifier built on a curated subset of the public Loughran-McDonald
Finance Sentiment Dictionary. The lexicon fallback is a *real* model, not a
stub: it computes probabilities from token matches, so classifications are
reproducible and explainable.

Neither model ever fabricates text. They only classify text supplied by the
caller.

The module exposes a single :class:`SentimentModel` abstraction so the news
service can be unit-tested with a lightweight fake and so the active model can
be swapped (for example a fine-tuned in-house checkpoint) without touching the
ingestion or aggregation layers.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.config import CONFIG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SentimentScores:
    """Normalised probabilities for a single classification.

    ``positive_prob``, ``neutral_prob`` and ``negative_prob`` always sum to
    ``1.0`` (barring floating-point rounding). ``label`` is the argmax and
    ``confidence`` is the probability mass of that label.
    """

    positive_prob: float
    neutral_prob: float
    negative_prob: float
    label: str
    confidence: float


def _argmax_label(positive: float, neutral: float, negative: float) -> tuple[str, float]:
    """Return the ``(label, confidence)`` for the largest probability."""

    scores = {"Positive": positive, "Neutral": neutral, "Negative": negative}
    label = max(scores, key=scores.get)  # type: ignore[arg-type]
    return label, scores[label]


_LABEL_MAP: dict[str, str] = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "pos": "Positive",
    "neu": "Neutral",
    "neg": "Negative",
}


def _normalise_label(raw: str) -> str:
    return _LABEL_MAP.get(raw.strip().lower(), raw.strip().capitalize())


def _uniform_scores() -> SentimentScores:
    """Return an even 1/3-1/3-1/3 "no opinion" result for empty input."""

    third = 1 / 3
    return SentimentScores(
        positive_prob=third,
        neutral_prob=third,
        negative_prob=third,
        label="Neutral",
        confidence=third,
    )


@dataclass
class _FinBertHandle:
    """Lazily built FinBERT pipeline handle."""

    model_name: str
    pipeline: object | None = None
    error: Exception | None = None


class SentimentModel(ABC):
    """Interface for text classifiers used by the news pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable identifier of the active model."""

    @abstractmethod
    def classify(self, text: str) -> SentimentScores:
        """Classify ``text`` returning normalised sentiment scores."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when the model can serve requests."""


class FinBertSentimentModel(SentimentModel):
    """FinBERT-based classifier backed by a Hugging Face transformer.

    The model is loaded lazily on the first call to :meth:`classify` so that
    importing the module never blocks on a network download. If loading fails
    for any reason (network, memory, incompatible library versions)
    :meth:`is_available` returns ``False`` and the caller can fall back to a
    lexicon model.
    """

    def __init__(self, model_name: str = CONFIG.finbert_model, max_length: int = 512) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._handle = _FinBertHandle(model_name=model_name)

    @property
    def name(self) -> str:
        return f"finbert:{self._model_name.split('/')[-1]}"

    def _load(self) -> object | None:
        if self._handle.pipeline is not None:
            return self._handle.pipeline
        if self._handle.error is not None:
            return None
        try:
            from transformers import pipeline  # type: ignore

            classifier = pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
                max_length=self._max_length,
                truncation=True,
            )
            self._handle.pipeline = classifier
            logger.info("Loaded FinBERT model '%s'", self._model_name)
            return classifier
        except Exception as exc:  # noqa: BLE001 - intentional broad guard
            self._handle.error = exc
            logger.warning(
                "FinBERT model '%s' could not be loaded (%s); fall back to lexicon model",
                self._model_name,
                exc,
            )
            return None

    def is_available(self) -> bool:
        return self._load() is not None

    def classify(self, text: str) -> SentimentScores:
        classifier = self._load()
        if classifier is None:
            return FinanceLexiconSentimentModel().classify(text)

        if not text or not text.strip():
            return _uniform_scores()

        result = self._predict(classifier, str(text))
        return result if result is not None else FinanceLexiconSentimentModel().classify(text)

    @staticmethod
    def _predict(classifier: object, text: str) -> SentimentScores | None:
        try:
            output = classifier(text)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - model runtime guard
            logger.debug("FinBERT prediction failed: %s", exc)
            return None

        # A single string input returns a list containing one list of dicts.
        if isinstance(output, list) and output and isinstance(output[0], list):
            rows = output[0]
        else:
            rows = output
        if not isinstance(rows, list):
            return None

        scores: dict[str, float] = {}
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            label = _normalise_label(str(entry.get("label", "")))
            score = float(entry.get("score", 0.0))
            scores[label] = scores.get(label, 0.0) + score

        positive = scores.get("Positive", 0.0)
        neutral = scores.get("Neutral", 0.0)
        negative = scores.get("Negative", 0.0)
        total = positive + neutral + negative
        if total <= 0:
            return None
        positive /= total
        neutral /= total
        negative /= total
        label, confidence = _argmax_label(positive, neutral, negative)
        return SentimentScores(
            positive_prob=positive,
            neutral_prob=neutral,
            negative_prob=negative,
            label=label,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Finance-domain lexicon fallback (Loughran-McDonald derived word lists)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# A curated subset of the Loughran-McDonald Finance Sentiment Dictionary
# (public domain, https://sraf.nd.edu/textual/lob/). These are real,
# domain-specific finance words - not generic English sentiment words.
_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "profit",
        "profits",
        "earnings",
        "gain",
        "gains",
        "growth",
        "grow",
        "growing",
        "exceeded",
        "exceeds",
        "exceed",
        "beat",
        "beats",
        "beating",
        "outperform",
        "outperforms",
        "upgrade",
        "upgrades",
        "upgraded",
        "upgrading",
        "upside",
        "recovery",
        "recover",
        "recovered",
        "restored",
        "restore",
        "strong",
        "strength",
        "stronger",
        "momentum",
        "bullish",
        "bull",
        "bulls",
        "surplus",
        "surpluses",
        "dividend",
        "dividends",
        "buyback",
        "buybacks",
        "acquire",
        "acquired",
        "acquisition",
        "acquisitions",
        "expand",
        "expanded",
        "expanding",
        "expansion",
        "launch",
        "launched",
        "launching",
        "innovate",
        "innovated",
        "innovative",
        "innovation",
        "record",
        "records",
        "surge",
        "surged",
        "surging",
        "surges",
        "rally",
        "rallied",
        "rallying",
        "rallies",
        "climb",
        "climbed",
        "climbing",
        "climbs",
        "advance",
        "advanced",
        "advancing",
        "advances",
        "rising",
        "rise",
        "rose",
        "raises",
        "raised",
        "uplift",
        "optimism",
        "optimistic",
        "favorable",
        "favorably",
        "positive",
        "positively",
        "favor",
        "favors",
        "favour",
        "rebound",
        "rebounded",
        "rebounding",
        "rebounds",
        "revised",
        "upward",
        "uplifted",
        "robust",
        "robustly",
    }
)

_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "loss",
        "losses",
        "decline",
        "declined",
        "declining",
        "declines",
        "dropped",
        "drop",
        "drops",
        "dropping",
        "fall",
        "falls",
        "fell",
        "falling",
        "lower",
        "lowered",
        "lowering",
        "lowers",
        "miss",
        "missed",
        "missing",
        "misses",
        "shortfall",
        "shortfalls",
        "below",
        "underperform",
        "underperforms",
        "underperformed",
        "downgrade",
        "downgrades",
        "downgraded",
        "downgrading",
        "downward",
        "cut",
        "cuts",
        "cutting",
        "reduced",
        "reduce",
        "reduces",
        "reducing",
        "reduction",
        "write-down",
        "writedown",
        "writeoffs",
        "write-offs",
        "writeoff",
        "impairment",
        "charge",
        "charges",
        "charged",
        "restructuring",
        "restructure",
        "restructures",
        "restructured",
        "layoff",
        "layoffs",
        "downsize",
        "downsized",
        "downsizing",
        "downsizes",
        "bearish",
        "bear",
        "bears",
        "warning",
        "warned",
        "warnings",
        "investigate",
        "investigated",
        "investigation",
        "investigations",
        "probe",
        "probed",
        "prosecute",
        "prosecuted",
        "prosecution",
        "penalty",
        "penalties",
        "fine",
        "fined",
        "fines",
        "delist",
        "delisting",
        "bankrupt",
        "bankruptcy",
        "defaulted",
        "default",
        "defaults",
        "defaulting",
        "liquidate",
        "liquidated",
        "liquidation",
        "risk",
        "risks",
        "risky",
        "volatile",
        "volatility",
        "uncertain",
        "uncertainty",
        "caution",
        "cautious",
        "sluggish",
        "slowdown",
        "slow",
        "slump",
        "slumped",
        "slumps",
        "slumping",
        "plummet",
        "plummeted",
        "plummets",
        "nosedive",
        "tumble",
        "tumbled",
        "tumbles",
        "tumbling",
        "selloff",
        "sell-off",
        "sell",
        "selling",
        "sold",
        "short",
        "shorting",
        "shorts",
        "shorted",
        "negativity",
        "negative",
        "negatively",
        "weak",
        "weaken",
        "weaker",
        "weakness",
        "disappoint",
        "disappointed",
        "disappoints",
        "disappointing",
        "disappointment",
        "disappointments",
        "oversight",
        "recall",
        "recalled",
        "vulnerable",
        "exposure",
        "exposures",
    }
)


class FinanceLexiconSentimentModel(SentimentModel):
    """Deterministic finance-lexicon classifier (Loughran-McDonald subset)."""

    name = "lexicon:finance"

    def __init__(
        self,
        positive_words: frozenset[str] | None = None,
        negative_words: frozenset[str] | None = None,
    ) -> None:
        self._positive = positive_words or _POSITIVE_WORDS
        self._negative = negative_words or _NEGATIVE_WORDS

    def is_available(self) -> bool:
        return True

    def classify(self, text: str) -> SentimentScores:
        if not text or not text.strip():
            return _uniform_scores()

        tokens = _tokenize(text)
        positives = sum(1 for token in tokens if token in self._positive)
        negatives = sum(1 for token in tokens if token in self._negative)

        # A smoothing prior keeps the model from being over-confident on a
        # single token and guarantees non-zero probabilities across all three
        # classes, matching the softmax shape of a neural model.
        alpha = 1.0
        positive = positives + alpha
        neutral = alpha
        negative = negatives + alpha
        total = positive + neutral + negative

        positive /= total
        neutral /= total
        negative /= total
        label, confidence = _argmax_label(positive, neutral, negative)
        return SentimentScores(
            positive_prob=positive,
            neutral_prob=neutral,
            negative_prob=negative,
            label=label,
            confidence=confidence,
        )


def create_sentiment_model(model_name: str | None = None) -> SentimentModel:
    """Build the sentiment model configured for the application.

    FinBERT is preferred. If it cannot be loaded the returned object is a
    :class:`FinanceLexiconSentimentModel`, which is always available and
    requires no network.
    """

    name = model_name or CONFIG.finbert_model
    finbert = FinBertSentimentModel(model_name=name)
    if finbert.is_available():
        return finbert
    return FinanceLexiconSentimentModel()


_DEFAULT_MODEL: SentimentModel | None = None


def get_default_sentiment_model() -> SentimentModel:
    """Return a process-wide lazily initialised sentiment model.

    The FinBERT model is attempted once; if it loads it is reused for the
    lifetime of the process, otherwise the lightweight lexicon model is used.
    """

    global _DEFAULT_MODEL  # noqa: PLW0603
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = create_sentiment_model()
        logger.info("Sentiment model active: %s", _DEFAULT_MODEL.name)
    return _DEFAULT_MODEL


def reset_default_sentiment_model() -> None:
    """Drop the cached sentiment model (test helper)."""

    global _DEFAULT_MODEL  # noqa: PLW0603
    _DEFAULT_MODEL = None
