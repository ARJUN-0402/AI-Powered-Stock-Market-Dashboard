"""Streamlit entrypoint for the AI-Powered Stock Market Dashboard.

This file is intentionally thin: it orchestrates layout and delegates every
business logic step to a service module under :mod:`src`. Keeping the UI
layer free of analytics means the same services can be reused from
notebooks, scripts or tests.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.alerts.alert_engine import (
    Alert,
    alerts_have_negative,
    alerts_have_positive,
    generate_alerts,
)
from src.config import CONFIG
from src.data.market_data_service import (
    InvalidSymbolError,
    MarketDataError,
    MarketDataService,
    SymbolNotFoundError,
    get_current_price,
    get_default_service,
)
from src.data.news_service import (
    AggregateSentiment,
    ArticleSentiment,
    NewsDataError,
    NewsDataService,
)
from src.data.news_service import (
    get_default_service as get_default_news_service,
)
from src.features.technical_indicators import (
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    latest_value,
)
from src.ml.explainability import explain_model, explain_prediction
from src.ml.prediction import PredictionResult, predict_from_history
from src.ml.preprocessing import (
    align_features_target,
    build_feature_frame,
    build_target,
    time_aware_split,
)
from src.ml.training import (
    ModelBundle,
    TrainingConfig,
    train_models,
)
from src.portfolio.portfolio_engine import Holding, summarise_price_frame
from src.utils.logging import configure_logging, get_logger
from src.utils.numeric import is_nan
from src.utils.validation import safe_float
from src.visualization.charts import (
    create_aggregate_sentiment_chart,
    create_atr_chart,
    create_bollinger_chart,
    create_candlestick_chart,
    create_drawdown_chart,
    create_macd_chart,
    create_mini_chart,
    create_obv_chart,
    create_rsi_chart,
    create_sentiment_breakdown_chart,
    create_sentiment_trend_chart,
    create_stochastic_chart,
    create_volatility_chart,
    create_volume_chart,
)

logger = get_logger(__name__)


_CSS = """
<style>
    :root {
        --background-color: #0d1117;
        --card-background: #161b22;
        --text-color: #c9d1d9;
        --accent-color: #58a6ff;
        --positive-color: #3fb950;
        --negative-color: #f85149;
        --border-color: #30363d;
    }

    body {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stApp {
        background-color: var(--background-color);
    }

    .css-1d391kg {
        background-color: var(--card-background);
    }

    .st-bx {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
    }

    .st-cs {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
    }

    .css-1offfwp {
        background-color: var(--card-background);
    }

    .stButton>button {
        background-color: var(--accent-color);
        color: white;
        border-radius: 4px;
    }

    .stSelectbox>div>div {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
    }

    .stDataFrame {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
    }

    .positive { color: var(--positive-color); font-weight: bold; }
    .negative { color: var(--negative-color); font-weight: bold; }

    .metric-card {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .alert-positive {
        background-color: rgba(63, 185, 80, 0.1);
        border: 1px solid var(--positive-color);
        border-radius: 4px;
        padding: 8px;
        margin: 4px 0;
    }

    .alert-negative {
        background-color: rgba(248, 81, 73, 0.1);
        border: 1px solid var(--negative-color);
        border-radius: 4px;
        padding: 8px;
        margin: 4px 0;
    }

    .news-item {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 8px;
    }

    .sentiment-positive { color: var(--positive-color); }
    .sentiment-negative { color: var(--negative-color); }
    .sentiment-neutral  { color: var(--text-color); }
</style>
"""


def _configure_page() -> None:
    """Configure the Streamlit page and inject custom CSS."""

    st.set_page_config(
        page_title=CONFIG.app_title,
        page_icon=CONFIG.page_icon,
        layout=CONFIG.layout,
        initial_sidebar_state=CONFIG.initial_sidebar_state,
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def _sidebar_controls() -> tuple[str, str, bool, bool]:
    """Render the sidebar and return the selected options."""

    st.sidebar.title("Dashboard Controls")

    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        sorted(CONFIG.watchlist),
        index=0,
    )

    time_period = st.sidebar.selectbox(
        "Select Time Period",
        sorted(CONFIG.valid_periods),
        index=sorted(CONFIG.valid_periods).index(CONFIG.default_period),
    )

    st.sidebar.subheader("Chart Settings")
    show_mas = st.sidebar.checkbox("Show Moving Averages", value=True)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)

    return selected_stock, time_period, show_mas, show_volume


def _price_metrics(data: pd.DataFrame, mas: dict) -> dict:
    """Compute headline metrics displayed in the dashboard."""

    current_price = safe_float(data["Close"].iloc[-1])
    previous_price = safe_float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price else 0.0

    rsi_series = calculate_rsi(data)
    current_rsi = latest_value(rsi_series)

    macd_series, _, _ = calculate_macd(data)
    current_macd = latest_value(macd_series)

    ma_50_series = mas.get("MA_50")
    current_ma_50 = latest_value(ma_50_series) if ma_50_series is not None else float("nan")

    return {
        "current_price": current_price,
        "previous_price": previous_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "current_rsi": current_rsi,
        "current_macd": current_macd,
        "current_ma_50": current_ma_50,
    }


def _render_metric_cards(symbol: str, metrics: dict) -> None:
    """Render the four headline metric cards."""

    cols = st.columns(4)

    with cols[0]:
        cls = "positive" if metrics["price_change"] >= 0 else "negative"
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{symbol}</h3>
                <h2>${metrics['current_price']:.2f}</h2>
                <p class="{cls}">
                    {metrics['price_change']:+.2f} ({metrics['price_change_pct']:+.2f}%)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        current_rsi = metrics["current_rsi"]
        if not is_nan(current_rsi):
            tone = (
                "positive"
                if current_rsi < CONFIG.rsi_oversold
                else "negative" if current_rsi > CONFIG.rsi_overbought else ""
            )
            label = (
                "Oversold"
                if current_rsi < CONFIG.rsi_oversold
                else "Overbought" if current_rsi > CONFIG.rsi_overbought else "Neutral"
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>RSI (14)</h3>
                    <h2>{current_rsi:.2f}</h2>
                    <p class="{tone}">{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with cols[2]:
        current_macd = metrics["current_macd"]
        if not is_nan(current_macd):
            tone = "positive" if current_macd > 0 else "negative"
            label = "Bullish" if current_macd > 0 else "Bearish"
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>MACD</h3>
                    <h2>{current_macd:.2f}</h2>
                    <p class="{tone}">{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with cols[3]:
        ma_50 = metrics["current_ma_50"]
        if not is_nan(ma_50):
            tone = "positive" if metrics["current_price"] > ma_50 else "negative"
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>MA (50)</h3>
                    <h2>${ma_50:.2f}</h2>
                    <p class="{tone}">{metrics['current_price'] - ma_50:+.2f} vs MA</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_alerts(symbol: str, alerts: list[Alert]) -> None:
    """Render the alerts section."""

    if not alerts:
        return
    st.subheader("🔔 Alerts")
    for alert in alerts:
        css_class = (
            "alert-positive"
            if alerts_have_positive([alert])
            else "alert-negative" if alerts_have_negative([alert]) else ""
        )
        st.markdown(
            f"<div class='{css_class}'>⚠️ {symbol}{alert.message}</div>",
            unsafe_allow_html=True,
        )


def _render_watchlist(selected: str, service: MarketDataService | None = None) -> list[Holding]:
    """Render the watchlist column and return the holdings for inspection."""

    st.subheader("Watchlist")
    watchlist_stocks = [
        s for s in sorted(CONFIG.watchlist)[: CONFIG.watchlist_size] if s != selected
    ]
    holdings: list[Holding] = []
    market_service = service or get_default_service()
    for stock in watchlist_stocks:
        try:
            watchlist_data = market_service.get_daily_history(stock, period="1mo")
        except (InvalidSymbolError, SymbolNotFoundError, MarketDataError) as exc:
            logger.warning("Skipping watchlist entry %s: %s", stock, exc)
            continue
        if watchlist_data.empty:
            continue
        holding = summarise_price_frame(stock, watchlist_data)
        holdings.append(holding)
        cls = "positive" if holding.change >= 0 else "negative"
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>{stock}</h4>
                <p>${holding.current_price:.2f}
                <span class="{cls}">({holding.change_pct:+.2f}%)</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(create_mini_chart(watchlist_data, stock), use_container_width=True)
    return holdings


def _sentiment_class(label: str) -> str:
    if label == "Positive":
        return "sentiment-positive"
    if label == "Negative":
        return "sentiment-negative"
    return "sentiment-neutral"


def _format_timestamp(dt: object) -> str:
    try:
        from datetime import datetime

        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if hasattr(dt, "astimezone"):
            local = dt.astimezone()  # type: ignore[attr-defined]
            return local.strftime("%b %d, %Y %H:%M %Z").strip()
    except Exception:  # noqa: BLE001 - formatting must never break the UI
        pass
    return "unknown time"


# ---------------------------------------------------------------------------
# ML Pipeline Integration
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Building features and target…")
def _build_ml_dataset(data: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe features and aligned target for a given symbol."""
    features = build_feature_frame(data, windows=(5, 20, 50))
    target = build_target(data, horizon=horizon, threshold=0.0)
    return align_features_target(features, target)


@st.cache_data(show_spinner="Preparing time-aware splits…")
def _time_aware_splits(
    features: pd.DataFrame,
    target: pd.Series,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    gap: int = 1,
    min_train: int = 20,
) -> tuple:
    """Return chronological train/validation/test splits with purge gaps."""
    return time_aware_split(
        features,
        target,
        train_fraction=train_frac,
        validation_fraction=val_frac,
        test_fraction=test_frac,
        gap_rows=gap,
        min_train_rows=min_train,
    )


@st.cache_resource(show_spinner="Training models…")
def _train_ml_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    config: TrainingConfig | None = None,
) -> tuple[dict[str, ModelBundle], dict, dict, str]:
    """Train and compare models, returning bundles, comparisons, split info, and selected name."""
    try:
        result = train_models(features, target, config=config)
        bundles = result.bundles
        comparisons = {
            name: {
                "validation_metrics": comp.validation_metrics,
                "test_metrics": comp.test_metrics,
                "baseline_metrics": comp.baseline_metrics,
                "selected": comp.selected,
            }
            for name, comp in result.comparisons.items()
        }
        return bundles, comparisons, result.split_info, result.selected_model_name
    except Exception as exc:  # noqa: BLE001 - surface gracefully in UI
        logger.exception("Model training failed: %s", exc)
        return {}, {}, {}, ""


@st.cache_data(show_spinner="Generating prediction…")
def _make_prediction(
    bundle: ModelBundle,
    data: pd.DataFrame,
    symbol: str,
) -> PredictionResult:
    """Predict direction for the latest available feature row."""
    return predict_from_history(bundle, data, symbol=symbol)


def _render_ml_section(
    symbol: str,
    data: pd.DataFrame,
    market_service: MarketDataService,
) -> None:
    """Render the educational ML prediction section with disclaimers and details."""
    st.markdown("---")
    st.subheader("🧪 Educational ML Prediction (Not Financial Advice)")

    # Explicit disclaimer
    st.warning(
        (
            "**⚠️ IMPORTANT DISCLAIMER**\n\n"
            "This prediction is **educational only** — it is **not a fact, "
            "recommendation, guarantee, or automated trading signal**.\n\n"
            "- The model predicts the *direction* of the next-day close return "
            "(up/down) using historical patterns only.\n"
            "- Probabilities are model outputs, **not** calibrated confidence "
            "intervals or guarantees.\n"
            "- Past performance does not predict future results. Markets are "
            "influenced by countless unpredictable factors.\n"
            "- **Do not make investment decisions based on this output.** "
            "Consult a qualified financial advisor."
        ),
        icon="⚠️",
    )

    # Check minimum history requirements
    min_rows = 60  # minimum for feature engineering + splits
    if len(data) < min_rows:
        st.info(
            f"Insufficient history for ML prediction: need at least {min_rows} daily rows, "
            f"got {len(data)}. Select a longer time period (e.g., 1y or 2y)."
        )
        return

    # Build dataset (cached)
    features, target = _build_ml_dataset(data)
    if features.empty or target.empty:
        st.info(
            "Unable to build ML dataset — not enough valid feature/target "
            "rows after alignment."
        )
        return

    # Train models (cached)
    bundles, comparisons, split_info, selected_name = _train_ml_models(features, target)

    if not bundles or not selected_name:
        st.info(
            "Model training did not produce a usable model. This can happen "
            "with limited data or class imbalance."
        )
        return

    selected_bundle = bundles[selected_name]
    metadata = selected_bundle.metadata

    # Prediction for latest row
    prediction = _make_prediction(selected_bundle, data, symbol)

    if not prediction.is_available:
        st.info("Prediction unavailable for the latest row — insufficient features or model error.")
        return

    # --- Display Prediction ---
    st.markdown("### 📊 Latest Prediction")

    pred_cols = st.columns([1, 1, 1, 1])
    with pred_cols[0]:
        direction_label = "📈 Up" if prediction.prediction == "up" else "📉 Down"
        st.metric("Predicted Direction", direction_label)
    with pred_cols[1]:
        st.metric("Probability (Winning Class)", f"{prediction.probability:.1%}")
    with pred_cols[2]:
        st.metric("Up Probability", f"{prediction.up_probability:.1%}")
    with pred_cols[3]:
        st.metric("Down Probability", f"{prediction.down_probability:.1%}")

    # Model metadata
    st.markdown("### 🤖 Model Details")
    meta_cols = st.columns([1, 1])
    with meta_cols[0]:
        st.caption(f"**Model:** {metadata.model_name}")
        st.caption(f"**Version:** {metadata.model_version}")
        st.caption(f"**Target:** {metadata.target_definition}")
    with meta_cols[1]:
        st.caption(f"**Trained:** {_format_timestamp(metadata.trained_at)}")
        st.caption(
            f"**Training Window:** {metadata.training_start} → "
            f"{metadata.training_end}"
        )
        st.caption(
            f"**Rows — Train/Val/Test:** {metadata.train_rows} / "
            f"{metadata.validation_rows} / {metadata.test_rows}"
        )

    # Validation vs Test comparison
    st.markdown("### 📈 Validation vs Test Performance")
    comp = comparisons[selected_name]
    val_metrics = comp["validation_metrics"]
    test_metrics = comp["test_metrics"]
    baseline = comp["baseline_metrics"]

    metrics_to_show = [
        ("Balanced Accuracy", "balanced_accuracy"),
        ("F1 Score", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("ROC-AUC", "roc_auc"),
    ]

    metric_rows = []
    for label, key in metrics_to_show:
        val = val_metrics.get(key)
        test = test_metrics.get(key)
        base = baseline.get(key)
        metric_rows.append({
            "Metric": label,
            "Validation": f"{val:.3f}" if val is not None else "N/A",
            "Test": f"{test:.3f}" if test is not None else "N/A",
            "Baseline (Majority Class)": f"{base:.3f}" if base is not None else "N/A",
        })

    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=False)

    st.caption(

            "Model is selected on **validation** performance only; test metrics "
            "are shown for transparency. "
            "Baseline is a majority-class classifier (predicts the most "
            "frequent class)."

    )

    # Top feature explanations
    st.markdown("### 🔍 Top Feature Explanations")
    st.caption(

            "Explanations show **association, not causation**. "
            "Tree importances are global (split frequency), not local "
            "contributions for this specific prediction."

    )

    try:
        # Get the latest transformed feature row
        transformed = selected_bundle.preprocessor.transform(features)
        latest_row = transformed.iloc[[-1]]
        feature_names = selected_bundle.feature_names

        # Get explanations
        explanations = explain_prediction(
            selected_bundle.estimator,
            feature_names,
            latest_row,
            top_n=10,
        )

        if explanations:
            expl_df = pd.DataFrame(explanations)
            # Show relevant columns
            display_cols = ["feature", "scope", "interpretation"]
            if "contribution" in expl_df.columns:
                display_cols.insert(1, "contribution")
            elif "importance" in expl_df.columns:
                display_cols.insert(1, "importance")

            st.dataframe(expl_df[display_cols], hide_index=True, use_container_width=False)
        else:
            st.info("No explanations available for this model type.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Explanation generation failed: %s", exc)
        st.info("Feature explanations unavailable for this model.")

    # Global model importance (fallback)
    st.markdown("#### Global Feature Importance (Training-Time)")
    try:
        global_explanations = explain_model(
            selected_bundle.estimator,
            selected_bundle.feature_names,
            top_n=10,
        )
        if global_explanations:
            global_df = pd.DataFrame(global_explanations)
            st.dataframe(global_df, hide_index=True, use_container_width=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Global explanation failed: %s", exc)

    # Split info
    with st.expander("📋 Split Details"):
        st.json(split_info)


def _article_to_chart_dict(item: ArticleSentiment) -> dict:
    return {
        "title": item.article.title,
        "label": item.label,
        "polarity": item.positive_prob - item.negative_prob,
        "positive_prob": item.positive_prob,
        "neutral_prob": item.neutral_prob,
        "negative_prob": item.negative_prob,
        "confidence": item.confidence,
    }


def _render_aggregate_sentiment(agg: AggregateSentiment) -> None:
    """Render an aggregated (market/window) sentiment block."""

    st.markdown(
        f"**Aggregate market sentiment ({agg.window}) — {agg.label}** "
        f"· confidence {agg.confidence:.0%} · {agg.article_count} article(s) analyzed"
    )
    st.caption(
        "Aggregate sentiment is the mean of per-article model probabilities, "
        "not a prediction of future price movement."
    )
    cols = st.columns(3)
    cols[0].metric("Positive", f"{agg.positive_prob:.0%}")
    cols[1].metric("Neutral", f"{agg.neutral_prob:.0%}")
    cols[2].metric("Negative", f"{agg.negative_prob:.0%}")
    st.plotly_chart(
        create_aggregate_sentiment_chart(
            agg.positive_prob,
            agg.neutral_prob,
            agg.negative_prob,
            label=agg.label,
        ),
        use_container_width=True,
    )


def _render_article(item: ArticleSentiment, service: NewsDataService) -> None:
    article = item.article
    cls = _sentiment_class(item.label)
    url = article.url or "#"
    ts = _format_timestamp(article.published_at)
    st.markdown(
        f"""
        <div class="news-item">
            <a href="{url}" target="_blank">{article.title}</a>
            <p><small>{article.source} · {ts} · analyzed by {item.model}</small></p>
            <p class="{cls}">{item.label} ({item.confidence:.0%}) —
            pos {item.positive_prob:.0%} · neu {item.neutral_prob:.0%} ·
            neg {item.negative_prob:.0%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_news(symbol: str, news_service: NewsDataService | None = None) -> None:
    """Render the real news sentiment panel."""

    st.subheader("News Sentiment")
    service = news_service or get_default_news_service()

    try:
        scored = service.get_news_with_sentiment(symbol)
    except InvalidSymbolError as exc:
        st.warning(f"{symbol} is not a valid ticker — news unavailable.")
        logger.info("News skipped for invalid symbol %s: %s", symbol, exc)
        return
    except NewsDataError as exc:
        logger.warning("News fetch failed for %s: %s", symbol, exc)
        st.info(
            "News is temporarily unavailable for this stock. "
            "This may be due to rate limiting, a missing API key, or a network "
            "issue. Prices and indicators are unaffected."
        )
        return
    except Exception as exc:  # noqa: BLE001 - never crash the dashboard
        logger.exception("Unexpected news error for %s: %s", symbol, exc)
        st.info("News is temporarily unavailable for this stock.")
        return

    if not scored:
        st.info("No recent news available for this stock.")
        return

    status = service.provider_status(len(scored))
    st.caption(
        f"Source: {status.name} · Model: {status.model} · "
        f"{len(scored)} article(s) analyzed"
    )

    try:
        aggregate = service.get_aggregate_sentiment(symbol, window="recent")
        _render_aggregate_sentiment(aggregate)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Aggregate sentiment unavailable: %s", exc)

    try:
        trend = service.get_news_trend(symbol, days=7)
        if trend:
            st.plotly_chart(
                create_sentiment_trend_chart(trend, symbol=symbol),
                use_container_width=True,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sentiment trend unavailable: %s", exc)

    st.plotly_chart(
        create_sentiment_breakdown_chart(
            [_article_to_chart_dict(item) for item in scored[: CONFIG.news_limit]]
        ),
        use_container_width=True,
    )

    st.caption(
        "Each dot is a real article. Sentiment is the model's classification "
        "of the headline/description text, not a price forecast."
    )
    for item in scored[: CONFIG.news_limit]:
        _render_article(item, service)


def main() -> None:
    """Application entrypoint."""

    configure_logging()
    _configure_page()

    st.markdown(
        f"<h1 style='text-align: center; color: #58a6ff;'>"
        f"{CONFIG.page_icon} {CONFIG.app_title}</h1>",
        unsafe_allow_html=True,
    )

    selected_stock, time_period, show_mas, show_volume = _sidebar_controls()

    market_service = get_default_service()
    try:
        data = market_service.get_history(selected_stock, period=time_period, interval="1d")
    except (InvalidSymbolError, SymbolNotFoundError, MarketDataError) as exc:
        logger.error("Unable to load data for %s: %s", selected_stock, exc)
        st.error(f"Unable to load data for {selected_stock}: {exc}")
        return
    if data is None or data.empty:
        st.error("No data available for the selected stock and time period.")
        return

    mas = calculate_moving_averages(data)
    metrics = _price_metrics(data, mas)
    alerts = generate_alerts(data, selected_stock)

    col_main, col_side = st.columns([3, 1])

    with col_main:
        _render_metric_cards(selected_stock, metrics)
        _render_alerts(selected_stock, alerts)

        st.subheader(f"{selected_stock} Price Chart")
        st.plotly_chart(
            create_candlestick_chart(data, selected_stock, show_mas, show_volume),
            use_container_width=True,
        )

        st.subheader("Technical Indicators")
        tab_rsi, tab_macd, tab_bb, tab_stoch, tab_vol = st.tabs(
            ["RSI", "MACD", "Bollinger", "Stochastic", "Volatility"]
        )
        with tab_rsi:
            st.plotly_chart(create_rsi_chart(data), use_container_width=True)
        with tab_macd:
            st.plotly_chart(create_macd_chart(data), use_container_width=True)
        with tab_bb:
            st.plotly_chart(create_bollinger_chart(data, selected_stock), use_container_width=True)
        with tab_stoch:
            st.plotly_chart(create_stochastic_chart(data), use_container_width=True)
        with tab_vol:
            st.plotly_chart(create_volatility_chart(data), use_container_width=True)

        st.subheader("Volume & Risk")
        tab_obv, tab_atr, tab_dd = st.tabs(["OBV", "ATR", "Drawdown"])
        with tab_obv:
            st.plotly_chart(create_obv_chart(data), use_container_width=True)
        with tab_atr:
            st.plotly_chart(create_atr_chart(data), use_container_width=True)
        with tab_dd:
            st.plotly_chart(create_drawdown_chart(data), use_container_width=True)

        if show_volume:
            st.subheader("Trading Volume")
            st.plotly_chart(create_volume_chart(data), use_container_width=True)

    with col_side:
        _render_watchlist(selected_stock, get_default_service())
        _render_news(selected_stock, get_default_news_service())

    # Educational ML Prediction Section
    _render_ml_section(selected_stock, data, market_service)

    # Live price is fetched but kept available for future widgets. Surfacing
    # it through the logger avoids changing the UI while keeping the call.
    live_price: float | None = get_current_price(selected_stock, service=market_service)
    if live_price is not None:
        logger.debug("Live price for %s: %.2f", selected_stock, live_price)


if __name__ == "__main__":
    main()
