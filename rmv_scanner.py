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

# Fetch stock data from Polygon.io with retries and debugging
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
                print(f"✅ {ticker}: {len(data['results'])} days retrieved")  # Debugging line
                return data["results"]
            else:
                print(f"⚠️ {ticker}: No data returned!")  # Debugging line
                return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {ticker}: {e}")
            time.sleep(2 ** attempt) 

    return []

# Compute RMV with debugging
def compute_rmv(data):
    if len(data) < 20:
        print("⚠️ Not enough data to calculate RMV")
        return 0  

    closes = np.array([entry["c"] for entry in data])
    highs = np.array([entry["h"] for entry in data])
    lows = np.array([entry["l"] for entry in data])

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]

    tr = np.maximum(highs - lows, np.maximum(abs(highs - prev_close), abs(lows - prev_close)))

    atr1 = np.mean(tr[-5:])
    atr2 = np.mean(tr[-10:])
    atr3 = np.mean(tr[-20:])

    avg_atr = np.mean([atr1, atr2, atr3])
    highest_atr = np.max([atr1, atr2, atr3])
    lowest_atr = np.min([atr1, atr2, atr3])

    print(f"🔍 ATR1: {atr1}, ATR2: {atr2}, ATR3: {atr3}")  # Debugging ATR values

    if highest_atr == lowest_atr:
        print("⚠️ ATR values are too close, RMV calculation skipped")
        return 0  

    rmv = ((avg_atr - lowest_atr) / (highest_atr - lowest_atr)) * 100
    print(f"📊 RMV Calculated: {rmv}")  # Debugging RMV value

    return round(rmv, 2)

# Streamlit UI
st.title("📉 RMV Stock Scanner")
uploaded_file = st.file_uploader("📂 Upload CSV with 'Ticker' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "Ticker" not in df.columns:
        st.error("⚠️ CSV must have a 'Ticker' column!")
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

        # Debugging: Show all RMV values
        print(results_df)

        # Filter RMV values below 20
        filtered_df = results_df[results_df["RMV"] < 20]

        # Display in Streamlit UI
        if not filtered_df.empty:
            st.write("📋 **Stocks with RMV below 20:**")
            st.dataframe(filtered_df)
        else:
            st.write("⚠️ No stocks found with RMV below 20.")

        # Download button for results
        st.download_button(
            label="⬇️ Download RMV Results",
            data=results_df.to_csv(index=False),
            file_name="rmv_results.csv",
            mime="text/csv"
        )

