"""Validation and normalisation helpers for the news data layer.

These helpers centralise the rules for deciding whether a raw article is
usable and for deduplicating, time-zone-normalising and recency-filtering
collections of articles. Providers are responsible only for the upstream
fetch; the service layer owns every shape/rate/sanity concern.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from src.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_TITLE_LEN = 300
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def parse_timestamp_utc(value: object) -> datetime | None:
    """Parse an arbitrary timestamp into a tz-aware UTC datetime.

    Supports ISO-8601 strings (with or without a trailing ``Z``), RFC-2822
    style strings (as produced by some RSS providers), numeric epoch seconds
    and :class:`datetime` / :class:`pandas.Timestamp` objects. Returns
    ``None`` when the value cannot be parsed.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, int | float):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        # Normalise a trailing "Z" (UTC Zulu) to an explicit offset so
        # ``fromisoformat`` accepts it on every Python version.
        normalised = text.replace("Z", "+00:00").strip()
        try:
            dt = datetime.fromisoformat(normalised)
        except ValueError:
            # Fall back to email utils for RFC-2822 timestamps.
            try:
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalise_title(value: object) -> str:
    """Strip HTML and collapse whitespace from a headline."""

    if value is None:
        return ""
    text = _HTML_TAG_RE.sub("", str(value))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > _MAX_TITLE_LEN:
        text = text[:_MAX_TITLE_LEN]
    return text


def normalise_description(value: object) -> str | None:
    """Strip HTML from a description, returning ``None`` when empty."""

    if value is None:
        return None
    text = _HTML_TAG_RE.sub("", str(value))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def validate_article(article: object) -> bool:
    """Return ``True`` when ``article`` is a :class:`NewsArticle` worth keeping.

    The check is deliberately cheap: it only rejects obvious junk (empty
    title, missing URL, missing source, no timestamp).
    """

    # Imported here to avoid a circular import at module import time.
    from src.data.news_service.dto import NewsArticle

    if not isinstance(article, NewsArticle):
        return False
    if not article.title.strip():
        return False
    if not article.source.strip():
        return False
    # A URL is expected for every real article; the dashboard links to it.
    if not article.url.strip():
        return False
    return article.published_at is not None and article.published_at.tzinfo is not None


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def deduplicate_articles(
    articles: Sequence,
    *,
    similarity_threshold: float = 0.9,
) -> list:
    """Remove duplicate and near-duplicate headlines.

    Articles are de-duplicated in two passes:

    1. **Exact URL** match — if two articles share a URL only the first is
       kept (handles the common case of multiple providers republishing the
       same story).
    2. **Title similarity** — using :class:`difflib.SequenceMatcher`,
       articles whose normalised titles exceed ``similarity_threshold`` are
       treated as the same story; the earliest published copy is retained.

    The input order is preserved so callers can rely on chronological or
    relevance ordering applied beforehand.
    """

    from src.data.news_service.dto import NewsArticle

    seen_urls: set[str] = set()
    kept: list[NewsArticle] = []

    for article in articles:
        if not isinstance(article, NewsArticle):
            continue
        url = article.url.strip().lower()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        kept.append(article)

    if similarity_threshold >= 1.0:
        return kept

    deduped: list[NewsArticle] = []
    for candidate in kept:
        dupe = False
        for existing in deduped:
            ratio = SequenceMatcher(
                None, _title_key(candidate.title), _title_key(existing.title)
            ).ratio()
            if ratio >= similarity_threshold:
                dupe = True
                break
        if not dupe:
            deduped.append(candidate)
    return deduped


def filter_recent_articles(
    articles: Sequence,
    *,
    hours: int = 72,
) -> list:
    """Drop articles published more than ``hours`` ago.

    Articles with a naive/missing timestamp were already rejected by
    :func:`validate_article`; here we keep only those within the window
    measured from the newest article's timestamp (so a back-filled feed is
    anchored to its freshest item rather than to wall-clock ``now``).
    """

    from src.data.news_service.dto import NewsArticle

    if not articles:
        return []
    typed = [a for a in articles if isinstance(a, NewsArticle)]
    if not typed:
        return []

    anchor = max(a.published_at for a in typed)
    cutoff = anchor - timedelta(hours=hours)
    return [a for a in typed if a.published_at >= cutoff]
