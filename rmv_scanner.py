import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
from datetime import datetime, timedelta

# Load API key securely
try:
    POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
except KeyError:
    st.error("API Key not found! Check secrets.toml")
    st.stop()

# Fetch stock data from Polygon.io with retries
def fetch_stock_data(ticker, timespan="day", limit=50, max_retries=3):
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=limit * 2)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
    
    params = {"apiKey": POLYGON_API_KEY, "limit": limit}
    session = requests.Session()

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data and data["results"]:
                return data["results"]
            else:
                return []  # No data
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)  # Exponential backoff

    return []

# Compute RMV
def compute_rmv(data, atr_periods=[5, 10, 20], lookback=50):
    if len(data) < max(atr_periods) + 1:  
        return 0  # Not enough data

    highs = np.array([entry["h"] for entry in data])
    lows = np.array([entry["l"] for entry in data])
    closes = np.array([entry["c"] for entry in data])

    # Ensure same-length close array for True Range calculations
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]  # Avoid using uninitialized value

    # Compute True Range
    tr = np.maximum(highs - lows, np.maximum(abs(highs - prev_closes), abs(lows - prev_closes)))

    # Compute ATR for different periods
    atr_values = []
    for period in atr_periods:
        if len(tr) >= period:
            atr_values.append(np.convolve(tr, np.ones(period)/period, mode='valid')[-1])

    if not atr_values:
        return 0  # No ATR values computed

    # Average the ATRs for RMV tightness calculation
    avg_atr = np.mean(atr_values)

    # Normalization over lookback period
    max_atr = max(atr_values[-lookback:]) if len(atr_values) >= lookback else max(atr_values)
    min_atr = min(atr_values[-lookback:]) if len(atr_values) >= lookback else min(atr_values)

    if max_atr == min_atr:
        return 0  # Avoid divide by zero

    rmv = ((avg_atr - min_atr) / (max_atr - min_atr)) * 100
    return round(rmv, 2)

# Streamlit UI
st.title("RMV Stock Scanner")
uploaded_file = st.file_uploader("Upload CSV with 'Ticker' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "Ticker" not in df.columns:
        st.error("CSV must have a 'Ticker' column!")
    else:
        tickers = df["Ticker"].tolist()
        results = []
        progress_bar = st.progress(0)
        total_stocks = len(tickers)

        for index, ticker in enumerate(tickers):
            stock_data = fetch_stock_data(ticker)
            rmv_value = compute_rmv(stock_data)
            results.append({"Ticker": ticker, "RMV": rmv_value})

            progress_bar.progress((index + 1) / total_stocks)
            time.sleep(0.6)  # Optimized for Polygon.io $29 plan

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)  # Display as table

        st.download_button(
            label="Download RMV Results",
            data=results_df.to_csv(index=False),
            file_name="rmv_results.csv",
            mime="text/csv"
        )


