"""Tests for the TTL cache used by the market data service."""

from __future__ import annotations

import time

from src.data.market_data_service.cache import TTLCache


def test_cache_returns_none_on_miss() -> None:
    cache: TTLCache[str] = TTLCache(default_ttl_seconds=10)
    assert cache.get("missing") is None


def test_cache_returns_value_on_hit() -> None:
    cache: TTLCache[str] = TTLCache(default_ttl_seconds=10)
    cache.set("k", "value")
    assert cache.get("k") == "value"
    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_cache_expires_entries() -> None:
    cache: TTLCache[str] = TTLCache(default_ttl_seconds=0.05)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.1)
    assert cache.get("k") is None
    assert cache.stats["evictions"] == 1


def test_cache_clear_removes_everything() -> None:
    cache: TTLCache[str] = TTLCache(default_ttl_seconds=10)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_cache_invalidate_single_key() -> None:
    cache: TTLCache[str] = TTLCache(default_ttl_seconds=10)
    cache.set("a", "1")
    cache.invalidate("a")
    assert cache.get("a") is None


def test_cache_honours_explicit_ttl() -> None:
    cache: TTLCache[str] = TTLCache(default_ttl_seconds=10)
    cache.set("k", "v", ttl_seconds=0.05)
    time.sleep(0.1)
    assert cache.get("k") is None
