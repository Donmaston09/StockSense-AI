import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime, timedelta
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyAT3S7MgtNGjd6O-ZzPejX22tuAs105MXU")

# ✅ Set page configuration
st.set_page_config(page_title="StockSense AI", layout="wide")

# StockSense AI Introduction
st.sidebar.title("🚀 StockSense AI")
st.sidebar.write("Your AI-powered financial analyst for smart investments! 📊")
st.sidebar.markdown(
    """
    🔍 **What it does:**
    ✅ **AI-Powered Insights** – Market overview, deep analysis
    ✅ **Real-Time Stock Data** – Prices, trends, fundamentals
    ✅ **Analyst Recommendations** – Expert ratings & insights
    ✅ **Market News & Sentiment** – Competitive analysis 📈📉
    ✅ **Risk Assessment** – Volatility, debt-to-equity, uncertainties
    """
)

def analyze_stock(ticker, last_close, month_change, pe_ratio, market_cap, eps):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')  # Use the correct method
        prompt = f"""
        You are a seasoned Wall Street analyst with deep expertise in market analysis! 📊

        Follow these steps for a comprehensive financial analysis:
        1. **Market Overview** – Latest stock price, 52-week high & low
        2. **Financial Deep Dive** – P/E ratio, Market Cap, EPS
        3. **Professional Insights** – Analyst recommendations, ratings
        4. **Market Context** – Industry trends, sentiment indicators

        **Your reporting style:**
        - Begin with an **executive summary**
        - Use **tables** for data presentation
        - Include **clear section headers**
        - Add **emoji indicators** for trends 📈📉
        - Highlight key insights with **bullet points**
        - Compare metrics to **industry averages**
        - Explain **technical terms**
        - End with **forward-looking analysis**

        **Risk Disclosure:**
        - Always highlight potential **risk factors**
        - Note **market uncertainties**
        - Mention relevant **regulatory concerns**

        """
        response = model.generate_content(prompt)  # Use generate_content instead of generate_text
        return response.text
    except Exception as e:
        return f"Error during analysis: {e}"

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    return stock, hist

def get_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=2f031206d3ba4eb98c63e384d53a873c"
    response = requests.get(url).json()
    articles = response.get("articles", [])[:5]
    return articles

def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity if text else 0

def get_analyst_ratings(stock):
    try:
        return stock.recommendations.tail(5)
    except:
        return None

def get_stock_performance(hist):
    last_close = hist["Close"].iloc[-1]
    one_month_ago = hist.index[-1] - timedelta(days=30)
    month_change = (last_close - hist.loc[hist.index >= one_month_ago, "Close"].iloc[0]) / hist.loc[hist.index >= one_month_ago, "Close"].iloc[0] * 100
    return last_close, month_change

def calculate_rsi(hist, period=14):
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(hist):
    short_ema = hist['Close'].ewm(span=12, adjust=False).mean()
    long_ema = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    return macd, macd.ewm(span=9, adjust=False).mean()

st.title("📈 StockSense AI")

# ✅ Function to create AI agent
def create_agent():
    return genai.GenerativeModel(model_name='gemini-1.5-flash')  # Use the correct method

# ✅ Initialize AI agent
agent = create_agent()

tickers = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA, GOOGL):", "AAPL, TSLA")
tickers = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]

for ticker in tickers:
    st.subheader(f"📌 {ticker} Analysis")
    stock, hist = get_stock_data(ticker)

    # Fetch stock metrics
    info = stock.info
    last_close, month_change = get_stock_performance(hist)
    pe_ratio = info.get("trailingPE", "N/A")
    market_cap = info.get("marketCap", "N/A")
    eps = info.get("trailingEps", "N/A")

    # Analyze stock using Gemini AI
    analysis = analyze_stock(ticker, last_close, month_change, pe_ratio, market_cap, eps)
    st.write(analysis)

    st.metric(label="Last Closing Price", value=f"${last_close:.2f}")
    st.metric(label="1-Month Change", value=f"{month_change:.2f}%", delta=month_change)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Closing Price"))
    fig.update_layout(title=f"{ticker} Stock Price Trend (Last 6 Months)", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig)

    hist["RSI"] = calculate_rsi(hist)
    hist["MACD"], hist["Signal"] = calculate_macd(hist)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], mode="lines", name="RSI"))
    fig_rsi.update_layout(title=f"{ticker} RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig_rsi)

    st.subheader("🔍 Analyst Insights")
    analyst_ratings = get_analyst_ratings(stock)
    st.dataframe(analyst_ratings) if analyst_ratings is not None else st.write("No recent analyst recommendations available.")

    st.subheader("📰 Market Context & News")
    news_articles = get_news(ticker)
    for article in news_articles:
        st.write(f"**[{article['title']}]({article['url']})**")
        sentiment = sentiment_analysis(article["description"])
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        st.write(f"Sentiment: **{sentiment_label}** ({sentiment:.2f})")
        st.write("---")

st.caption("Built with ❤️ using Streamlit, Gemini AI & Yahoo Finance API")
