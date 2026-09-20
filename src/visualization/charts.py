"""Chart factories used by the Streamlit UI.

These helpers build :mod:`plotly` figures. They contain no Streamlit
dependencies and can be unit-tested independently.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.config import CONFIG
from src.features.analytics import (
    atr,
    bollinger_bands,
    drawdown,
    rolling_volatility,
    stochastic_oscillator,
)
from src.features.technical_indicators import (
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
)

_DARK_TEMPLATE = "plotly_dark"


def create_candlestick_chart(
    data: pd.DataFrame,
    symbol: str,
    show_mas: bool = True,
    show_volume: bool = True,
) -> go.Figure:
    """Build a candlestick chart with optional moving averages and volume."""

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol,
        )
    )

    if show_mas:
        mas = calculate_moving_averages(data)
        for label, ma in mas.items():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma,
                    mode="lines",
                    name=label,
                    line=dict(width=1),
                )
            )

    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template=_DARK_TEMPLATE,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Volume is currently rendered in a separate chart, but we keep the
    # parameter for backward compatibility.
    _ = show_volume
    return fig


def create_volume_chart(data: pd.DataFrame) -> go.Figure:
    """Build a coloured volume bar chart."""

    fig = go.Figure()
    colors = [
        "green" if close >= open_ else "red"
        for close, open_ in zip(data["Close"], data["Open"], strict=False)
    ]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            marker_color=colors,
            name="Volume",
        )
    )
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=200,
        template=_DARK_TEMPLATE,
    )
    return fig


def create_rsi_chart(data: pd.DataFrame) -> go.Figure:
    """Build a stand-alone RSI chart."""

    rsi = calculate_rsi(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rsi,
            mode="lines",
            name="RSI",
            line=dict(color="blue"),
        )
    )
    fig.add_hline(
        y=CONFIG.rsi_overbought,
        line_dash="dash",
        line_color="red",
        annotation_text="Overbought",
    )
    fig.add_hline(
        y=CONFIG.rsi_oversold,
        line_dash="dash",
        line_color="green",
        annotation_text="Oversold",
    )
    fig.add_hline(y=50, line_dash="dot", line_color="white")
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=300,
        template=_DARK_TEMPLATE,
        yaxis_range=[0, 100],
    )
    return fig


def create_macd_chart(data: pd.DataFrame) -> go.Figure:
    """Build a stand-alone MACD chart."""

    macd, signal, histogram = calculate_macd(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.index, y=macd, mode="lines", name="MACD", line=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signal, mode="lines", name="Signal", line=dict(color="orange"))
    )
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=histogram,
            name="Histogram",
            marker_color=["green" if val >= 0 else "red" for val in histogram],
        )
    )
    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Date",
        yaxis_title="Value",
        height=300,
        template=_DARK_TEMPLATE,
    )
    return fig


def create_mini_chart(
    data: pd.DataFrame,
    symbol: str,
    window: int = CONFIG.mini_chart_window,
) -> go.Figure:
    """Build a compact candlestick used in the watchlist."""

    window = max(1, min(window, len(data)))
    tail = data.tail(window)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["Open"],
            high=tail["High"],
            low=tail["Low"],
            close=tail["Close"],
            name=symbol,
        )
    )
    fig.update_layout(
        title=f"{symbol}",
        height=150,
        template=_DARK_TEMPLATE,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def create_sentiment_bar(sentiments: Iterable[dict[str, object]]) -> go.Figure:
    """Build a bar chart of news sentiment polarities."""

    titles = [str(item.get("title", ""))[:30] for item in sentiments]
    polarities = [float(item.get("polarity", 0.0)) for item in sentiments]

    fig = go.Figure(
        go.Bar(
            x=titles,
            y=polarities,
            marker_color=["green" if p > 0 else "red" if p < 0 else "grey" for p in polarities],
        )
    )
    fig.update_layout(
        title="News Sentiment Polarity",
        height=250,
        template=_DARK_TEMPLATE,
        yaxis_title="Polarity",
    )
    return fig


def empty_figure(message: str | None = None) -> go.Figure:
    """Return an empty figure with a friendly message."""

    text = message or "No data available"
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template=_DARK_TEMPLATE, height=200)
    return fig


def create_bollinger_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Build a Bollinger Bands chart overlaid on price."""

    bb = bollinger_bands(data)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=bb["upper"],
            mode="lines",
            name="Upper Band",
            line=dict(width=1, color="rgba(255,255,255,0.5)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=bb["middle"],
            mode="lines",
            name="Middle (SMA 20)",
            line=dict(width=1, color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=bb["lower"],
            mode="lines",
            name="Lower Band",
            line=dict(width=1, color="rgba(255,255,255,0.5)"),
            fill="tonexty",
            fillcolor="rgba(88,166,255,0.1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=symbol,
            line=dict(width=2, color="#58a6ff"),
        )
    )
    fig.update_layout(
        title=f"{symbol} Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        template=_DARK_TEMPLATE,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_atr_chart(data: pd.DataFrame) -> go.Figure:
    """Build an Average True Range chart."""

    atr_values = atr(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=atr_values,
            mode="lines",
            name="ATR",
            line=dict(color="purple"),
        )
    )
    fig.update_layout(
        title="Average True Range (ATR)",
        xaxis_title="Date",
        yaxis_title="ATR ($)",
        height=250,
        template=_DARK_TEMPLATE,
    )
    return fig


def create_stochastic_chart(data: pd.DataFrame) -> go.Figure:
    """Build a Stochastic Oscillator chart."""

    stoch = stochastic_oscillator(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=stoch["%K"],
            mode="lines",
            name="%K",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=stoch["%D"],
            mode="lines",
            name="%D",
            line=dict(color="orange"),
        )
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(
        title="Stochastic Oscillator",
        xaxis_title="Date",
        yaxis_title="Value",
        height=250,
        template=_DARK_TEMPLATE,
        yaxis_range=[0, 100],
    )
    return fig


def create_volatility_chart(data: pd.DataFrame) -> go.Figure:
    """Build a rolling volatility chart."""

    vol = rolling_volatility(data, annualize=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=vol * 100,
            mode="lines",
            name="Annualized Volatility",
            line=dict(color="yellow"),
        )
    )
    fig.update_layout(
        title="Rolling Volatility (Annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=250,
        template=_DARK_TEMPLATE,
    )
    return fig


def create_drawdown_chart(data: pd.DataFrame) -> go.Figure:
    """Build a drawdown chart."""

    dd = drawdown(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=dd * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color="red"),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.2)",
        )
    )
    fig.update_layout(
        title="Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=250,
        template=_DARK_TEMPLATE,
    )
    return fig


def create_obv_chart(data: pd.DataFrame) -> go.Figure:
    """Build an On-Balance Volume chart."""

    from src.features.analytics import obv

    obv_values = obv(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=obv_values,
            mode="lines",
            name="OBV",
            line=dict(color="cyan"),
        )
    )
    fig.update_layout(
        title="On-Balance Volume (OBV)",
        xaxis_title="Date",
        yaxis_title="OBV",
        height=250,
        template=_DARK_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Sentiment visualisations
# ---------------------------------------------------------------------------

_SENTIMENT_COLORS = {
    "Positive": "#3fb950",
    "Negative": "#f85149",
    "Neutral": "#c9d1d9",
}


def _sentiment_color(label: str) -> str:
    return _SENTIMENT_COLORS.get(label, "#c9d1d9")


def create_sentiment_trend_chart(
    points: Sequence[dict[str, Any]],
    *,
    symbol: str = "",
) -> go.Figure:
    """Build a net-sentiment trend line over daily buckets.

    Each point is expected to expose ``date`` (ISO ``YYYY-MM-DD``),
    ``sentiment`` (net polarity in ``[-1, 1]``) and ``count`` (number of
    articles). An empty input yields a friendly "no data" figure.
    """

    fig = go.Figure()
    if not points:
        return empty_figure("No sentiment trend data available")

    dates = [p["date"] for p in points]
    net = [float(p.get("sentiment", 0.0)) for p in points]
    counts = [int(p.get("count", 0)) for p in points]

    line_color = "#3fb950" if net[-1] >= 0 else "#f85149"
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=net,
            mode="lines+markers",
            name="Net sentiment",
            line=dict(color=line_color, width=2),
            marker=dict(size=[max(4, c) for c in counts], color=counts, colorscale="Blues"),
            text=[f"{d} — {c} article(s)" for d, c in zip(dates, counts, strict=True)],
            hovertemplate="%{text}<br>Net sentiment: %{y:+.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="white")
    fig.update_layout(
        title=f"{symbol} News Sentiment Trend" if symbol else "News Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Net sentiment (pos − neg)",
        height=260,
        template=_DARK_TEMPLATE,
        yaxis_range=[-1, 1],
        showlegend=False,
    )
    return fig


def create_sentiment_breakdown_chart(
    items: Sequence[dict[str, Any]],
) -> go.Figure:
    """Build a per-article net-polarity bar chart.

    Each item is expected to expose ``title``, ``label`` and
    ``positive_prob`` / ``negative_prob`` (or a ``polarity`` value). Bars are
    coloured by the predicted label.
    """

    fig = go.Figure()
    if not items:
        return empty_figure("No article sentiment to display")

    titles = [str(item.get("title", ""))[:50] for item in items]
    colors: list[str] = []
    values: list[float] = []
    hover_text: list[str] = []
    for item in items:
        if "positive_prob" in item and "negative_prob" in item:
            polarity = float(item["positive_prob"]) - float(item["negative_prob"])
        else:
            polarity = float(item.get("polarity", 0.0))
        label = str(item.get("label", ""))
        colors.append(_sentiment_color(label))
        values.append(polarity)
        hover_text.append(f"{label} — {item.get('confidence', 0):.2f}")

    fig.add_trace(
        go.Bar(
            x=titles,
            y=values,
            marker_color=colors,
            text=hover_text,
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            name="Article sentiment",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="white")
    fig.update_layout(
        title="Per-article Sentiment",
        xaxis_title="Article",
        yaxis_title="Net polarity (pos − neg)",
        height=320,
        template=_DARK_TEMPLATE,
        yaxis_range=[-1, 1],
        showlegend=False,
    )
    return fig


def create_aggregate_sentiment_chart(
    positive_prob: float,
    neutral_prob: float,
    negative_prob: float,
    *,
    label: str = "",
) -> go.Figure:
    """Build a stacked/distribution chart of an aggregate sentiment.

    Accepts the three class probabilities directly (rather than a DTO) so the
    helper stays decoupled from the news data layer and is trivial to test.
    """

    values = [positive_prob, neutral_prob, negative_prob]
    colors = ["#3fb950", "#c9d1d9", "#f85149"]
    fig = go.Figure(
        go.Bar(
            x=["Positive", "Neutral", "Negative"],
            y=values,
            marker_color=colors,
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            showlegend=False,
        )
    )
    title = f"Aggregate sentiment — {label}" if label else "Aggregate sentiment"
    fig.update_layout(
        title=title,
        xaxis_title="Sentiment",
        yaxis_title="Probability",
        height=200,
        template=_DARK_TEMPLATE,
        yaxis_range=[0, 1],
    )
    return fig
