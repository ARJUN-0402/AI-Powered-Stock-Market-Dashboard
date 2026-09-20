"""Integration tests covering composed services."""

from __future__ import annotations

from src.alerts.alert_engine import generate_alerts
from src.features.technical_indicators import calculate_macd, calculate_rsi
from src.nlp.sentiment import build_news_items
from src.portfolio.portfolio_engine import summarise_price_frame
from tests.fixtures import make_price_frame


def test_end_to_end_pipeline_on_synthetic_frame() -> None:
    data = make_price_frame(rows=120)
    rsi = calculate_rsi(data)
    macd, signal, histogram = calculate_macd(data)
    alerts = generate_alerts(data, "AAPL")
    holding = summarise_price_frame("AAPL", data)
    news = build_news_items(["Apple posts record revenue", "Concerns over margins"])

    assert len(rsi) == len(data)
    assert len(macd) == len(signal) == len(histogram)
    # We do not assert alert content because the fixture may or may not
    # trigger thresholds, but the call must always return a list.
    assert isinstance(alerts, list)
    assert holding.symbol == "AAPL"
    assert len(news) == 2


def test_alerts_never_raise_on_pathological_inputs() -> None:
    import pandas as pd

    assert generate_alerts(pd.DataFrame(), "X") == []
    assert generate_alerts(None, "X") == []  # type: ignore[arg-type]
