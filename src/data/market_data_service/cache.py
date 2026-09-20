"""Thread-safe in-process TTL cache for the market data service.

The cache lives at the service layer (not the provider) so that
multiple services can share the same underlying provider and so that
test doubles do not need to reimplement caching logic. The cache is
intentionally simple: a single mutex guards a dict mapping cache
keys to ``(expires_at_monotonic, expires_at_wallclock, value)``
tuples.

* ``monotonic`` is used for expiry checks so clock skew cannot keep
  stale data alive forever.
* ``wallclock`` is exposed for observability and tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

V = TypeVar("V")


@dataclass(frozen=True)
class CacheEntry(Generic[V]):
    """A cached value with its expiry metadata."""

    value: V
    expires_at: float
    created_at: float

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class TTLCache(Generic[V]):
    """Tiny TTL cache with optional per-entry negative caching.

    Negative caching (storing the "no data" result) is useful to
    prevent repeated calls when a ticker is invalid. It can be
    disabled by passing ``negative_ttl_seconds=None``.
    """

    def __init__(
        self,
        *,
        default_ttl_seconds: float = 60.0,
        negative_ttl_seconds: float | None = 30.0,
    ) -> None:
        self._entries: dict[str, CacheEntry[V]] = {}
        self._lock = Lock()
        self._default_ttl = float(default_ttl_seconds)
        self._negative_ttl = (
            float(negative_ttl_seconds) if negative_ttl_seconds is not None else None
        )
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                self._evictions += 1
                self._entries.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: V, *, ttl_seconds: float | None = None) -> None:
        ttl = self._ttl_for(value, ttl_seconds)
        if ttl is None or ttl <= 0:
            return
        now = time.monotonic()
        entry: CacheEntry[V] = CacheEntry(
            value=value,
            expires_at=now + ttl,
            created_at=time.time(),
        )
        with self._lock:
            self._entries[key] = entry

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        """Remove every entry whose key starts with ``prefix``."""

        with self._lock:
            for key in list(self._entries):
                if key.startswith(prefix):
                    self._entries.pop(key, None)
                    self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _ttl_for(self, value: Any, ttl_seconds: float | None) -> float | None:
        if ttl_seconds is not None:
            return float(ttl_seconds)
        is_negative = isinstance(value, _NEGATIVE_SENTINELS)
        if is_negative:
            if self._negative_ttl is None:
                return None
            return self._negative_ttl
        return self._default_ttl


# Sentinel tuple used to identify "no data" responses. Importing
# classes directly would create a circular import; the service layer
# populates this at runtime.
_NEGATIVE_SENTINELS: tuple[type, ...] = ()


def register_negative_sentinel(sentinel_type: type) -> None:
    """Register a class that should be cached for only ``negative_ttl``."""

    global _NEGATIVE_SENTINELS  # noqa: PLW0603
    if sentinel_type not in _NEGATIVE_SENTINELS:
        _NEGATIVE_SENTINELS = (*_NEGATIVE_SENTINELS, sentinel_type)
