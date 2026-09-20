"""Tests for the alert engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alerts.alert_engine import (
    AlertSeverity,
    alerts_have_negative,
    alerts_have_positive,
    generate_alerts,
)


def _build_frame(prices: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [100_000] * len(prices),
        },
        index=index,
    )


def test_generate_alerts_empty_data_returns_empty() -> None:
    assert generate_alerts(pd.DataFrame(), "AAPL") == []
    assert generate_alerts(None, "AAPL") == []  # type: ignore[arg-type]


def test_rsi_oversold_alert() -> None:
    prices = list(np.linspace(100, 60, 60))
    data = _build_frame(prices)
    alerts = generate_alerts(data, "TEST")
    assert any(alert.severity == AlertSeverity.POSITIVE for alert in alerts)
    assert alerts_have_positive(alerts)


def test_price_cross_below_ma_triggers_negative_alert() -> None:
    # Build a series where the previous close is above MA_5 and the current
    # close drops below MA_5. We need: prev_price > prev_ma AND
    # current_price < current_ma.
    # First, an uptrend so prices are above MA_5; then a sudden drop below.
    prices = [100 + i * 1.0 for i in range(20)] + [50]
    data = _build_frame(prices)
    alerts = generate_alerts(data, "TEST")
    assert any(alert.severity == AlertSeverity.NEGATIVE for alert in alerts)
    assert alerts_have_negative(alerts)


def test_no_alerts_on_flat_series() -> None:
    data = _build_frame([100.0] * 80)
    assert generate_alerts(data, "TEST") == []
