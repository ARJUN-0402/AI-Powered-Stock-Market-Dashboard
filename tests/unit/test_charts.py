"""Tests for chart factories."""

from __future__ import annotations

import plotly.graph_objects as go

from src.visualization.charts import (
    create_candlestick_chart,
    create_macd_chart,
    create_mini_chart,
    create_rsi_chart,
    create_volume_chart,
    empty_figure,
)
from tests.fixtures import make_price_frame


def test_candlestick_chart_returns_figure() -> None:
    data = make_price_frame(rows=60)
    fig = create_candlestick_chart(data, "AAPL", show_mas=True)
    assert isinstance(fig, go.Figure)
    # Candlestick + 3 moving averages => 4 traces.
    assert len(fig.data) == 4


def test_candlestick_chart_without_mas() -> None:
    data = make_price_frame(rows=60)
    fig = create_candlestick_chart(data, "AAPL", show_mas=False)
    assert len(fig.data) == 1


def test_volume_chart_uses_correct_colors() -> None:
    data = make_price_frame(rows=20)
    fig = create_volume_chart(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_rsi_and_macd_charts_build() -> None:
    data = make_price_frame(rows=80)
    assert isinstance(create_rsi_chart(data), go.Figure)
    assert isinstance(create_macd_chart(data), go.Figure)


def test_mini_chart_uses_window() -> None:
    data = make_price_frame(rows=80)
    fig = create_mini_chart(data, "AAPL", window=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_empty_figure_helper() -> None:
    fig = empty_figure("no data")
    assert isinstance(fig, go.Figure)
    assert any("no data" in str(a.text) for a in fig.layout.annotations)
