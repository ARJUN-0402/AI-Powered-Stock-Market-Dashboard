"""Centralised news data service.

:class:`NewsDataService` is the single entry point used by the rest of the
application to obtain ticker-specific news, per-article financial sentiment
and aggregated sentiment snapshots. It coordinates:

* Input validation (ticker symbols).
* Provider invocation (delegated to a :class:`NewsProvider`).
* Normalisation, deduplication and recency filtering of raw articles.
* Sentiment classification via a :class:`SentimentModel`.
* Aggregation into daily / weekly / recent windows.
* Caching (in-process thread-safe TTL cache).
* Error translation (provider exceptions are normalised into the typed
  hierarchy in :mod:`src.data.news_service.errors`).

The service never fabricates news or sentiment: a failure to retrieve data is
surfaced either as a typed exception or as an empty aggregate with
``article_count == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

from src.config import CONFIG
from src.data.market_data_service.cache import TTLCache
from src.data.news_service.aggregation import (
    aggregate_daily,
    aggregate_recent,
    aggregate_weekly,
)
from src.data.news_service.dto import (
    AggregateSentiment,
    ArticleSentiment,
    NewsArticle,
    ProviderStatus,
)
from src.data.news_service.errors import (
    InvalidSymbolError,
    NewsDataError,
    NewsProviderError,
)
from src.data.news_service.providers.base import NewsProvider
from src.data.news_service.providers.yfinance_news import YFinanceNewsProvider
from src.data.news_service.validators import (
    deduplicate_articles,
    filter_recent_articles,
    validate_article,
)
from src.nlp.sentiment_model import SentimentModel
from src.utils.logging import get_logger
from src.utils.validation import ValidationError, validate_symbol

logger = get_logger(__name__)


@dataclass(frozen=True)
class NewsDataServiceConfig:
    """Runtime configuration for the news service."""

    ttl_seconds: float = field(default=float(CONFIG.news_cache_ttl_seconds))
    negative_ttl_seconds: float = 60.0
    limit: int = field(default_factory=lambda: CONFIG.news_limit)
    recency_hours: int = field(default_factory=lambda: CONFIG.news_recency_hours)
    dedup_similarity: float = field(default_factory=lambda: CONFIG.news_dedup_similarity)


_WINDOW_FUNCS = {
    "daily": aggregate_daily,
    "weekly": aggregate_weekly,
    "recent": aggregate_recent,
}


def resolve_provider(name: str | None = None) -> NewsProvider:
    """Build a provider from its configuration name.

    ``yfinance`` (the default) needs no credentials. ``newsapi`` requires
    ``NEWSAPI_KEY``; if the key is missing the provider still constructs but
    will raise :class:`NewsAuthenticationError` on use, allowing the service
    to degrade clearly.
    """

    provider_name = (name or CONFIG.news_provider or "yfinance").lower()
    if provider_name == "yfinance":
        return YFinanceNewsProvider()
    if provider_name == "newsapi":
        from src.data.news_service.providers.newsapi_provider import NewsAPIProvider

        return NewsAPIProvider()
    raise NewsProviderError(f"Unknown news provider '{provider_name}'")


class NewsDataService:
    """Reusable, provider-agnostic news + sentiment orchestrator."""

    def __init__(
        self,
        provider: NewsProvider | None = None,
        sentiment_model: SentimentModel | None = None,
        *,
        config: NewsDataServiceConfig | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        self._provider: NewsProvider = provider or resolve_provider()
        # The sentiment model is resolved lazily on first use so that simply
        # constructing the service (or the process-wide singleton) never
        # triggers a potentially slow network model download.
        self._sentiment_model: SentimentModel | None = sentiment_model
        self._config = config or NewsDataServiceConfig()
        self._cache: TTLCache = cache or TTLCache(
            default_ttl_seconds=self._config.ttl_seconds,
            negative_ttl_seconds=self._config.negative_ttl_seconds,
        )

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model_name(self) -> str:
        return self._resolve_sentiment_model().name

    @property
    def cache_stats(self) -> dict[str, int]:
        return self._cache.stats

    def provider_status(self, article_count: int = 0) -> ProviderStatus:
        """Return attribution metadata so the UI can label the data source."""

        return ProviderStatus(
            name=self._provider.name,
            model=self._resolve_sentiment_model().name,
            is_real=True,
            article_count=article_count,
        )

    def configure(
        self,
        *,
        ttl_seconds: float | None = None,
        limit: int | None = None,
        recency_hours: int | None = None,
        dedup_similarity: float | None = None,
    ) -> None:
        """Mutate service configuration in-place (affects future cache keys)."""

        updates: dict[str, Any] = {**self._config.__dict__}
        if ttl_seconds is not None:
            updates["ttl_seconds"] = float(ttl_seconds)
        if limit is not None:
            updates["limit"] = int(limit)
        if recency_hours is not None:
            updates["recency_hours"] = int(recency_hours)
        if dedup_similarity is not None:
            updates["dedup_similarity"] = float(dedup_similarity)
        self._config = NewsDataServiceConfig(**updates)

    def refresh(self, symbol: str | None = None) -> None:
        """Invalidate cached entries for ``symbol`` (or the whole cache)."""

        if symbol is None:
            self._cache.clear()
            logger.debug("News cache cleared")
            return
        symbol_key = self._safe_symbol_key(symbol)
        if symbol_key is None:
            return
        for prefix in ("articles", "sentiment", "aggregate"):
            self._cache.invalidate_prefix(f"{prefix}:{symbol_key}:")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_articles(
        self,
        symbol: str,
        *,
        use_cache: bool = True,
        limit: int | None = None,
    ) -> list[NewsArticle]:
        """Return normalised, de-duplicated, recent articles for ``symbol``."""

        normalised = self._validate_symbol(symbol)
        article_limit = limit if limit is not None else self._config.limit
        cache_key = f"articles:{normalised}:{article_limit}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, list):
                logger.debug("Cache hit for %s", cache_key)
                return list(cached)

        logger.info(
            "Fetching news for %s (provider=%s, limit=%d)",
            normalised,
            self._provider.name,
            article_limit,
        )
        try:
            raw = self._provider.fetch_news(normalised, limit=article_limit)
        except NewsDataError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected news provider error for %s: %s", normalised, exc)
            raise NewsProviderError(
                f"Failed to fetch news for '{normalised}'"
            ) from exc

        articles = self._normalise_collection(list(raw), symbol=normalised)
        self._cache.set(cache_key, articles)
        return list(articles)

    def get_news_with_sentiment(
        self,
        symbol: str,
        *,
        use_cache: bool = True,
        limit: int | None = None,
    ) -> list[ArticleSentiment]:
        """Return articles enriched with per-article sentiment scores."""

        normalised = self._validate_symbol(symbol)
        article_limit = limit if limit is not None else self._config.limit
        cache_key = f"sentiment:{normalised}:{article_limit}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, list):
                logger.debug("Cache hit for %s", cache_key)
                return list(cached)

        articles = self.fetch_articles(symbol, use_cache=use_cache, limit=article_limit)
        scored = self._classify(articles)
        self._cache.set(cache_key, scored)
        return scored

    def get_aggregate_sentiment(
        self,
        symbol: str,
        *,
        window: str = "recent",
        hours: int | None = None,
        use_cache: bool = True,
    ) -> AggregateSentiment:
        """Return aggregated sentiment for ``symbol`` over ``window``."""

        normalised = self._validate_symbol(symbol)
        window = window.lower()
        if window not in _WINDOW_FUNCS:
            raise NewsDataError(
                f"Unknown aggregation window '{window}'. Allowed: {sorted(_WINDOW_FUNCS)}"
            )
        cache_key = f"aggregate:{normalised}:{window}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, AggregateSentiment):
                logger.debug("Cache hit for %s", cache_key)
                return cached

        scored = self.get_news_with_sentiment(
            symbol, use_cache=use_cache
        )
        hours = hours if hours is not None else self._config.recency_hours
        if window == "recent":
            aggregate = aggregate_recent(
                scored, symbol=normalised, hours=hours,
                provider=self._provider.name, model=self._sentiment_model.name,
            )
        else:
            aggregate = _WINDOW_FUNCS[window](
                scored, symbol=normalised,
                provider=self._provider.name, model=self._sentiment_model.name,
            )
        self._cache.set(cache_key, aggregate)
        return aggregate

    def get_news_trend(
        self,
        symbol: str,
        *,
        days: int = 7,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Return daily sentiment trend points for the last ``days`` days.

        Each point is ``{"date": iso, "label": str, "sentiment": float,
        "count": int}`` where ``sentiment`` is the net polarity
        (``positive - negative``) so the chart is signed.
        """

        scored = self.get_news_with_sentiment(symbol, use_cache=use_cache)
        if not scored:
            return []
        buckets: dict[str, list[ArticleSentiment]] = {}
        for item in scored:
            day = item.article.published_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
            buckets.setdefault(day, []).append(item)

        points: list[dict[str, Any]] = []
        for day in sorted(buckets):
            items = buckets[day]
            pos = sum(i.positive_prob for i in items) / len(items)
            neg = sum(i.negative_prob for i in items) / len(items)
            top = max(items, key=lambda i: i.confidence)
            points.append(
                {
                    "date": day,
                    "label": top.label,
                    "sentiment": round(pos - neg, 4),
                    "count": len(items),
                    "model": top.model,
                }
            )
        return points[-days:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sentiment_model(self) -> SentimentModel:
        """Return the active sentiment model, resolving it lazily if needed."""

        if self._sentiment_model is None:
            from src.nlp.sentiment_model import get_default_sentiment_model

            self._sentiment_model = get_default_sentiment_model()
        return self._sentiment_model

    def _validate_symbol(self, symbol: str) -> str:
        try:
            return validate_symbol(symbol)
        except ValidationError as exc:
            raise InvalidSymbolError(str(exc)) from exc

    @staticmethod
    def _safe_symbol_key(symbol: str) -> str | None:
        try:
            return validate_symbol(symbol)
        except ValidationError:
            return None

    def _normalise_collection(
        self, articles: list[NewsArticle], *, symbol: str
    ) -> list[NewsArticle]:
        valid = [a for a in articles if validate_article(a)]
        deduped = deduplicate_articles(
            valid, similarity_threshold=self._config.dedup_similarity
        )
        recent = filter_recent_articles(deduped, hours=self._config.recency_hours)
        recent.sort(key=lambda a: a.published_at, reverse=True)
        if len(deduped) > len(recent):
            logger.debug(
                "Filtered %d old articles for %s", len(deduped) - len(recent), symbol
            )
        return recent

    def _classify(self, articles: list[NewsArticle]) -> list[ArticleSentiment]:
        model = self._resolve_sentiment_model()
        scored: list[ArticleSentiment] = []
        for article in articles:
            text = _sentiment_text(article)
            try:
                scores = model.classify(text)
            except Exception as exc:  # noqa: BLE001 - per-article guard
                logger.debug("Sentiment failed for article '%s': %s", article.title, exc)
                continue
            scored.append(
                ArticleSentiment(
                    article=article,
                    positive_prob=scores.positive_prob,
                    neutral_prob=scores.neutral_prob,
                    negative_prob=scores.negative_prob,
                    label=scores.label,
                    confidence=scores.confidence,
                    model=model.name,
                )
            )
        return scored


def _sentiment_text(article: NewsArticle) -> str:
    parts = [article.title]
    if article.description:
        parts.append(article.description)
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Process-wide singleton helpers
# ---------------------------------------------------------------------------

_DEFAULT_SERVICE: NewsDataService | None = None


def get_default_service() -> NewsDataService:
    """Return a lazily constructed process-wide news service instance."""

    global _DEFAULT_SERVICE  # noqa: PLW0603
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = NewsDataService()
    return _DEFAULT_SERVICE


def reset_default_service() -> None:
    """Drop the process-wide news service instance (test helper)."""

    global _DEFAULT_SERVICE  # noqa: PLW0603
    _DEFAULT_SERVICE = None
