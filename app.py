import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import requests
from textblob import TextBlob
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data for TextBlob
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Set page config for wide layout
st.set_page_config(
    page_title="AI-Powered Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode trading terminal style
st.markdown("""
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
    
    .positive {
        color: var(--positive-color);
        font-weight: bold;
    }
    
    .negative {
        color: var(--negative-color);
        font-weight: bold;
    }
    
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
    
    .sentiment-positive {
        color: var(--positive-color);
    }
    
    .sentiment-negative {
        color: var(--negative-color);
    }
    
    .sentiment-neutral {
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for the given data.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): Period for RSI calculation (default: 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the given data.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        short_window (int): Short period for EMA (default: 12)
        long_window (int): Long period for EMA (default: 26)
        signal_window (int): Signal line period (default: 9)
    
    Returns:
        tuple: MACD line, Signal line, and Histogram
    """
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# Function to calculate Moving Averages
def calculate_moving_averages(data, windows=[5, 20, 50]):
    """
    Calculate Moving Averages for the given data.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        windows (list): List of window sizes for moving averages
    
    Returns:
        dict: Dictionary of moving averages
    """
    mas = {}
    for window in windows:
        mas[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return mas

# Function to fetch stock data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period="1mo", interval="1d"):
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol
        period (str): Time period for data
        interval (str): Data interval
    
    Returns:
        pd.DataFrame: Stock data
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# Function to fetch live price
@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_live_price(symbol):
    """
    Fetch live stock price from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        float: Current stock price
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        return None

# Function to fetch news sentiment
def fetch_news_sentiment(symbol):
    """
    Fetch and analyze news sentiment for a given stock.
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        list: List of news items with sentiment analysis
    """
    # This is a simplified version - in a real app, you would use a news API
    # For demo purposes, we'll simulate news based on stock performance
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        company_name = info.get('longName', symbol)
        
        # Simulate news articles
        news_articles = [
            f"{company_name} announces strong quarterly earnings beating analyst expectations",
            f"{company_name} faces regulatory challenges in key markets",
            f"Analysts upgrade {company_name} stock after product innovation",
            f"{company_name} expands into new international markets",
            f"Supply chain issues affect {company_name} production"
        ]
        
        sentiments = []
        for article in news_articles:
            # Analyze sentiment using TextBlob
            blob = TextBlob(str(article))  # Convert to string to avoid cached_property issue
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            sentiments.append({
                "title": article,
                "sentiment": sentiment,
                "polarity": polarity
            })
        
        return sentiments
    except Exception as e:
        return []

# Function to check for alerts
def check_alerts(data, symbol):
    """
    Check for trading alerts based on technical indicators.
    
    Args:
        data (pd.DataFrame): Stock data
        symbol (str): Stock symbol
    
    Returns:
        list: List of alert messages
    """
    alerts = []
    
    # Check RSI alerts
    if len(data) >= 14:
        rsi = calculate_rsi(data)
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            alerts.append(f" oversold (RSI: {current_rsi:.2f})")
        elif current_rsi > 70:
            alerts.append(f" overbought (RSI: {current_rsi:.2f})")
    
    # Check Moving Average crossovers
    if len(data) >= 50:
        mas = calculate_moving_averages(data)
        current_price = data['Close'].iloc[-1]
        ma_5 = mas['MA_5'].iloc[-1]
        ma_20 = mas['MA_20'].iloc[-1]
        
        # Check if price crossed above or below MA
        prev_price = data['Close'].iloc[-2]
        prev_ma_5 = mas['MA_5'].iloc[-2]
        
        if prev_price < prev_ma_5 and current_price > ma_5:
            alerts.append(f" price crossed above 5-day MA (${ma_5:.2f})")
        elif prev_price > prev_ma_5 and current_price < ma_5:
            alerts.append(f" price crossed below 5-day MA (${ma_5:.2f})")
    
    return alerts

# Function to create candlestick chart
def create_candlestick_chart(data, symbol, show_mas=True, show_volume=True):
    """
    Create a candlestick chart with optional moving averages.
    
    Args:
        data (pd.DataFrame): Stock data
        symbol (str): Stock symbol
        show_mas (bool): Whether to show moving averages
        show_volume (bool): Whether to show volume
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ))
    
    # Add moving averages if enabled
    if show_mas:
        mas = calculate_moving_averages(data)
        for key, ma in mas.items():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma,
                mode='lines',
                name=key,
                line=dict(width=1)
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Function to create volume chart
def create_volume_chart(data):
    """
    Create a volume chart.
    
    Args:
        data (pd.DataFrame): Stock data
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Color volume bars based on price movement
    colors = ['green' if close > open else 'red' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        name="Volume"
    ))
    
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=200,
        template="plotly_dark"
    )
    
    return fig

# Function to create RSI chart
def create_rsi_chart(data):
    """
    Create an RSI chart.
    
    Args:
        data (pd.DataFrame): Stock data
    
    Returns:
        go.Figure: Plotly figure object
    """
    rsi = calculate_rsi(data)
    
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='blue')
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="white")
    
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=300,
        template="plotly_dark",
        yaxis_range=[0, 100]
    )
    
    return fig

# Function to create MACD chart
def create_macd_chart(data):
    """
    Create a MACD chart.
    
    Args:
        data (pd.DataFrame): Stock data
    
    Returns:
        go.Figure: Plotly figure object
    """
    macd, signal, histogram = calculate_macd(data)
    
    fig = go.Figure()
    
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=macd,
        mode='lines',
        name='MACD',
        line=dict(color='blue')
    ))
    
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color='orange')
    ))
    
    # Add Histogram
    fig.add_trace(go.Bar(
        x=data.index,
        y=histogram,
        name='Histogram',
        marker_color=['green' if val >= 0 else 'red' for val in histogram]
    ))
    
    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Date",
        yaxis_title="Value",
        height=300,
        template="plotly_dark"
    )
    
    return fig

# Function to create mini candlestick chart for watchlist
def create_mini_chart(data, symbol):
    """
    Create a mini candlestick chart for the watchlist.
    
    Args:
        data (pd.DataFrame): Stock data
        symbol (str): Stock symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index[-30:],  # Last 30 data points
        open=data['Open'][-30:],
        high=data['High'][-30:],
        low=data['Low'][-30:],
        close=data['Close'][-30:],
        name=symbol
    ))
    
    fig.update_layout(
        title=f"{symbol}",
        height=150,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

# Main app
def main():
    """
    Main application function.
    """
    # App title
    st.markdown("<h1 style='text-align: center; color: #58a6ff;'>📈 AI-Powered Stock Market Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Extended stock list
    default_stocks = [
        'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'PYPL',
        'INTC', 'AMD', 'CRM', 'DIS', 'BA', 'JPM', 'V', 'JNJ', 'WMT', 'PG',
        'MA', 'UNH', 'HD', 'BAC', 'VZ', 'XOM', 'KO', 'PFE', 'T', 'MRK'
    ]
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        default_stocks,
        index=0
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        index=2
    )
    
    # Chart controls
    st.sidebar.subheader("Chart Settings")
    show_mas = st.sidebar.checkbox("Show Moving Averages", value=True)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    
    # Fetch data
    data = fetch_stock_data(selected_stock, period=time_period if time_period else "1mo")
    
    if data.empty:
        st.error("No data available for the selected stock and time period.")
        return
    
    # Calculate indicators
    rsi = calculate_rsi(data)
    macd, signal, histogram = calculate_macd(data)
    mas = calculate_moving_averages(data)
    
    # Check for alerts
    alerts = check_alerts(data, selected_stock)
    
    # Main dashboard layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Stock metrics
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{selected_stock}</h3>
                <h2>${current_price:.2f}</h2>
                <p class="{'positive' if price_change >= 0 else 'negative'}">
                    {price_change:+.2f} ({price_change_pct:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            if len(rsi) > 0:
                current_rsi = rsi.iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>RSI (14)</h3>
                    <h2>{current_rsi:.2f}</h2>
                    <p class="{'positive' if current_rsi < 30 else 'negative' if current_rsi > 70 else ''}">
                        {'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with metric_col3:
            if len(macd) > 0:
                current_macd = macd.iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>MACD</h3>
                    <h2>{current_macd:.2f}</h2>
                    <p class="{'positive' if current_macd > 0 else 'negative'}">
                        {'Bullish' if current_macd > 0 else 'Bearish'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with metric_col4:
            if 'MA_50' in mas and len(mas['MA_50']) > 0:
                ma_50 = mas['MA_50'].iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>MA (50)</h3>
                    <h2>${ma_50:.2f}</h2>
                    <p class="{'positive' if current_price > ma_50 else 'negative'}">
                        {current_price - ma_50:+.2f} vs MA
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Alerts section
        if alerts:
            st.subheader("🔔 Alerts")
            for alert in alerts:
                if "oversold" in alert or "crossed above" in alert:
                    st.markdown(f"<div class='alert-positive'>⚠️ {selected_stock}{alert}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert-negative'>⚠️ {selected_stock}{alert}</div>", unsafe_allow_html=True)
        
        # Main price chart
        st.subheader(f"{selected_stock} Price Chart")
        price_chart = create_candlestick_chart(data, selected_stock, show_mas, show_volume)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Technical indicators charts
        st.subheader("Technical Indicators")
        tab1, tab2 = st.tabs(["RSI", "MACD"])
        
        with tab1:
            rsi_chart = create_rsi_chart(data)
            st.plotly_chart(rsi_chart, use_container_width=True)
        
        with tab2:
            macd_chart = create_macd_chart(data)
            st.plotly_chart(macd_chart, use_container_width=True)
        
        # Volume chart
        if show_volume:
            st.subheader("Trading Volume")
            volume_chart = create_volume_chart(data)
            st.plotly_chart(volume_chart, use_container_width=True)
    
    with col2:
        # Watchlist
        st.subheader("Watchlist")
        watchlist_stocks = [s for s in default_stocks[:10] if s != selected_stock]  # Show top 10 stocks
        
        for stock in watchlist_stocks:
            watchlist_data = fetch_stock_data(stock, period="1mo")
            if not watchlist_data.empty:
                current = watchlist_data['Close'].iloc[-1]
                previous = watchlist_data['Close'].iloc[-2] if len(watchlist_data) > 1 else current
                change = current - previous
                change_pct = (change / previous) * 100 if previous != 0 else 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{stock}</h4>
                    <p>${current:.2f} <span class="{'positive' if change >= 0 else 'negative'}">({change_pct:+.2f}%)</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mini chart
                mini_chart = create_mini_chart(watchlist_data, stock)
                st.plotly_chart(mini_chart, use_container_width=True)
        
        # News sentiment
        st.subheader("News Sentiment")
        news_sentiments = fetch_news_sentiment(selected_stock)
        
        if news_sentiments:
            for news in news_sentiments[:5]:  # Show top 5 news
                sentiment_class = "sentiment-positive" if news['sentiment'] == "Positive" else \
                                 "sentiment-negative" if news['sentiment'] == "Negative" else "sentiment-neutral"
                
                st.markdown(f"""
                <div class="news-item">
                    <p>{news['title']}</p>
                    <p class="{sentiment_class}">Sentiment: {news['sentiment']} ({news['polarity']:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No news available for this stock.")

if __name__ == "__main__":
    main()