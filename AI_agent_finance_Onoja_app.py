import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")  # Get 6 months of data
    return stock, hist

# Function to fetch news data (using dynamic API key input)
def get_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url).json()
    articles = response.get("articles", [])[:5]  # Get top 5 articles
    return articles

# Function to analyze sentiment
def sentiment_analysis(text):
    if text:
        return TextBlob(text).sentiment.polarity
    return 0

# Function to fetch analyst recommendations
def get_analyst_ratings(stock):
    try:
        return stock.recommendations.tail(5)
    except:
        return None

# Function to fetch stock performance summary
def get_stock_performance(hist):
    last_close = hist["Close"].iloc[-1]
    one_month_ago = hist.index[-1] - timedelta(days=30)
    month_change = (last_close - hist.loc[hist.index >= one_month_ago, "Close"].iloc[0]) / hist.loc[hist.index >= one_month_ago, "Close"].iloc[0] * 100
    return last_close, month_change

# Function to calculate RSI
def calculate_rsi(hist, period=14):
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(hist):
    short_ema = hist['Close'].ewm(span=12, adjust=False).mean()
    long_ema = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Streamlit App UI
st.set_page_config(page_title="Finance AI Agent", layout="wide")

st.title("📈 StockSense AI Agent")

# User input for the API key
api_key = st.text_input("Enter your News API Key:", "")

# Multi-stock comparison input
tickers = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA, GOOGL):", "AAPL, TSLA")

tickers = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]

if api_key:  # Ensure API key is entered before proceeding
    for ticker in tickers:
        st.subheader(f"📌 {ticker} Analysis")

        stock, hist = get_stock_data(ticker)

        # EXECUTIVE SUMMARY
        st.write(f"**Company:** {stock.info.get('longName', 'N/A')}")
        st.write(f"**Industry:** {stock.info.get('industry', 'N/A')}")
        st.write(f"**Sector:** {stock.info.get('sector', 'N/A')}")
        st.write(f"**Market Cap:** {stock.info.get('marketCap', 'N/A'):,}")
        st.write(f"**P/E Ratio:** {stock.info.get('trailingPE', 'N/A')}")
        st.write(f"**Earnings Growth:** {stock.info.get('earningsGrowth', 'N/A')}")
        st.write(f"**Dividend Yield:** {stock.info.get('dividendYield', 'N/A')}")

        # STOCK PERFORMANCE
        last_close, month_change = get_stock_performance(hist)
        st.metric(label="Last Closing Price", value=f"${last_close:.2f}")
        st.metric(label="1-Month Change", value=f"{month_change:.2f}%", delta=month_change)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Closing Price"))
        fig.update_layout(title=f"{ticker} Stock Price Trend (Last 6 Months)", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig)

        # TECHNICAL INDICATORS
        st.subheader("📊 Technical Indicators")

        # RSI
        hist["RSI"] = calculate_rsi(hist)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title=f"{ticker} RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI")
        st.plotly_chart(fig_rsi)

        # MACD
        hist["MACD"], hist["Signal"] = calculate_macd(hist)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], mode="lines", name="MACD", line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["Signal"], mode="lines", name="Signal Line", line=dict(color='red')))
        fig_macd.update_layout(title=f"{ticker} MACD Indicator", xaxis_title="Date", yaxis_title="MACD")
        st.plotly_chart(fig_macd)

        # ANALYST INSIGHTS
        st.subheader("🔍 Analyst Insights")
        analyst_ratings = get_analyst_ratings(stock)
        if analyst_ratings is not None:
            st.dataframe(analyst_ratings)
        else:
            st.write("No recent analyst recommendations available.")

        # MARKET CONTEXT & NEWS
        st.subheader("📰 Market Context & News")
        news_articles = get_news(ticker, api_key)  # Pass the API key here
        for article in news_articles:
            st.write(f"**[{article['title']}]({article['url']})**")
            sentiment = sentiment_analysis(article["description"])
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            st.write(f"Sentiment: **{sentiment_label}** ({sentiment:.2f})")
            st.write("---")

        # RISK ASSESSMENT
        st.subheader("⚠️ Risk Assessment")
        beta = stock.info.get("beta", "N/A")
        debt_to_equity = stock.info.get("debtToEquity", "N/A")
        st.write(f"**Beta (Volatility):** {beta}")
        st.write(f"**Debt-to-Equity Ratio:** {debt_to_equity}")

        if beta != "N/A":
            if beta > 1:
                st.warning("This stock is more volatile than the market.")
            elif beta < 1:
                st.success("This stock is less volatile than the market.")

    st.caption("Built with ❤️ using Streamlit & Yahoo Finance API")
else:
    st.warning("Please enter your News API key to access market news.")
