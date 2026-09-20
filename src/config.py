"""Centralized application configuration.

This module centralises runtime configuration for the dashboard. Values can be
overridden via environment variables, which keeps secrets and deployment-time
settings out of the source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_env(name: str, default: str) -> str:
    """Return the environment variable value or a default fallback."""

    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _get_env_int(name: str, default: int) -> int:
    """Return an integer environment variable, falling back to ``default``."""

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_env_float(name: str, default: float) -> float:
    """Return a float environment variable, falling back to ``default``."""

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration container."""

    app_title: str = "AI-Powered Stock Market Dashboard"
    page_icon: str = "📈"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"

    default_period: str = "1mo"
    default_interval: str = "1d"
    valid_periods: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
        )
    )
    valid_intervals: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "1m",
                "2m",
                "5m",
                "15m",
                "30m",
                "60m",
                "90m",
                "1h",
                "1d",
                "5d",
                "1wk",
                "1mo",
                "3mo",
            }
        )
    )

    watchlist: frozenset[str] = field(
        default_factory=lambda: frozenset(
            [
                "AAPL",
                "MSFT",
                "TSLA",
                "GOOGL",
                "AMZN",
                "META",
                "NVDA",
                "NFLX",
                "ADBE",
                "PYPL",
                "INTC",
                "AMD",
                "CRM",
                "DIS",
                "BA",
                "JPM",
                "V",
                "JNJ",
                "WMT",
                "PG",
                "MA",
                "UNH",
                "HD",
                "BAC",
                "VZ",
                "XOM",
                "KO",
                "PFE",
                "T",
                "MRK",
            ]
        )
    )

    cache_ttl_seconds: int = 300
    live_price_ttl_seconds: int = 60
    news_cache_ttl_seconds: int = 600

    news_provider: str = field(
        default_factory=lambda: _get_env("NEWS_PROVIDER", "yfinance")
    )
    news_limit: int = field(default_factory=lambda: _get_env_int("NEWS_LIMIT", 20))
    finbert_model: str = field(
        default_factory=lambda: _get_env("FINBERT_MODEL", "yiyanghkust/finbert-tone")
    )
    sentiment_confidence_threshold: float = field(
        default_factory=lambda: _get_env_float("SENTIMENT_CONFIDENCE_THRESHOLD", 0.6)
    )
    news_dedup_similarity: float = field(
        default_factory=lambda: _get_env_float("NEWS_DEDUP_SIMILARITY", 0.9)
    )
    news_recency_hours: int = field(
        default_factory=lambda: _get_env_int("NEWS_RECENCY_HOURS", 72)
    )

    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_short: int = 12
    macd_long: int = 26
    macd_signal: int = 9
    moving_average_windows: frozenset[int] = field(default_factory=lambda: frozenset({5, 20, 50}))

    sentiment_positive_threshold: float = 0.1
    sentiment_negative_threshold: float = -0.1

    watchlist_size: int = 10
    mini_chart_window: int = 30

    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))


CONFIG = AppConfig()
