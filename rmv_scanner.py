import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
import os
from datetime import datetime, timedelta

# Set up Streamlit UI
st.title("📈 RMV Scanner - Upload Your TradingView Watchlist")

# Upload file
uploaded_file = st.file_uploader("Upload your TradingView Watchlist CSV", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file containing your watchlist (must have a 'Ticker' column).")
    st.stop()

# Read the CSV file
df = pd.read_csv(uploaded_file)

# Validate CSV format
if "Ticker" not in df.columns:
    st.error("🚨 CSV file must contain a 'Ticker' column.")
    st.stop()

tickers = df["Ticker"].dropna().tolist()

if not tickers:
    st.error("🚨 No tickers found in the CSV file. Please check your file.")
    st.stop()

st.success(f"✅ Loaded {len(tickers)} tickers.")

# API Key (Use Streamlit secrets for security)
POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]

# Polygon.io API setup
BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=120&apiKey=" + POLYGON_API_KEY

# Date range for data retrieval
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')  # Get last 120 days of data

def fetch_stock_data(ticker):
    """Fetches historical stock data from Polygon.io"""
    url = BASE_URL.format(ticker=ticker, start_date=start_date, end_date=end_date)
    response = requests.get(url)
    
    if response.status_code != 200:
        return None  # API failed for this ticker

    data = response.json()
    
    if "results" not in data or not isinstance(data["results"], list):
        return None  # No valid data received

    return data["results"]

def calculate_atr(data, period=14):
    """Calculates the Average True Range (ATR) for a given stock"""
    if len(data) < period:
        return None  # Not enough data

    highs = np.array([day["h"] for day in data])
    lows = np.array([day["l"] for day in data])
    closes = np.array([day["c"] for day in data])
    
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, shift=1))
    tr3 = np.abs(lows - np.roll(closes, shift=1))
    
    tr = np.maximum(np.maximum(tr1, tr2), tr3)[1:]  # Ignore first row
    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    
    return atr[-1] if len(atr) > 0 else None  # Return latest ATR

# Processing tickers
results = []
progress_bar = st.progress(0)

for i, ticker in enumerate(tickers):
    st.write(f"📊 Processing {ticker} ({i+1}/{len(tickers)})...")
    
    stock_data = fetch_stock_data(ticker)
    
    if stock_data is None:
        st.warning(f"⚠️ No data for {ticker}, skipping.")
        continue

    atr = calculate_atr(stock_data)
    
    if atr is None:
        st.warning(f"⚠️ ATR not calculated for {ticker}, skipping.")
        continue

    latest_close = stock_data[-1]["c"]
    
    if latest_close == 0:
        st.warning(f"⚠️ Invalid closing price (0) for {ticker}, skipping.")
        continue

    rmv = (atr / latest_close) * 100
    results.append({"Ticker": ticker, "Latest_RMV": rmv})

    # Update progress bar
    progress_bar.progress((i + 1) / len(tickers))

    time.sleep(1)  # API rate limit handling

# Display results
if results:
    output_df = pd.DataFrame(results)
    st.write("✅ RMV Calculation Completed! Results:")
    st.dataframe(output_df)

    # Allow CSV download
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download RMV Results", data=csv, file_name="rmv_results.csv", mime="text/csv")
else:
    st.error("🚨 No valid RMV data calculated.")

