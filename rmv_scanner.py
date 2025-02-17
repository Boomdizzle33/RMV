import os
import time
import pandas as pd
import streamlit as st
from polygon import RESTClient
import numpy as np

# Load API Key from Streamlit Secrets
try:
    POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
except KeyError:
    st.error("❌ API Key not found in Streamlit secrets! Check secrets.toml structure.")
    st.stop()

# Set up Polygon.io Client
client = RESTClient(POLYGON_API_KEY)

# CSV File Import
CSV_FILE = "watchlist.csv"  # Change this if your file has a different name

if not os.path.exists(CSV_FILE):
    st.error(f"❌ Error: {CSV_FILE} not found. Make sure the file is in the correct location.")
    st.stop()

# Load tickers from CSV
try:
    stock_list = pd.read_csv(CSV_FILE)
    if "Ticker" not in stock_list.columns:
        st.error("❌ Error: CSV file must contain a 'Ticker' column.")
        st.stop()
    tickers = stock_list["Ticker"].tolist()
except Exception as e:
    st.error(f"❌ Error loading CSV file: {e}")
    st.stop()

# Parameters for RMV Calculation
LOOKBACK_DAYS = 20  # Number of days to calculate RMV
VOLATILITY_WINDOW = 10  # RMV window
RMV_THRESHOLD = 20  # RMV filter level

def fetch_historical_data(ticker):
    """Fetches the last LOOKBACK_DAYS + VOLATILITY_WINDOW days of data for RMV calculation."""
    try:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK_DAYS + VOLATILITY_WINDOW)).strftime('%Y-%m-%d')

        bars = client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_=start_date, to=end_date, limit=LOOKBACK_DAYS + VOLATILITY_WINDOW)
        
        data = [{"date": bar.timestamp, "close": bar.close} for bar in bars]
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_rmv(df):
    """Calculates Relative Measured Volatility (RMV)."""
    if len(df) < VOLATILITY_WINDOW:
        return 0  # Not enough data

    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_std"] = df["log_returns"].rolling(window=VOLATILITY_WINDOW).std()
    
    rmv = df["rolling_std"].iloc[-1] * 100
    return round(rmv, 2) if not np.isnan(rmv) else 0

# Process Stocks and Calculate RMV
results = []

progress_bar = st.progress(0)
total_stocks = len(tickers)

for i, ticker in enumerate(tickers):
    df = fetch_historical_data(ticker)
    
    if df.empty:
        st.warning(f"⚠️ Skipping {ticker}, no data available.")
        continue

    rmv_value = calculate_rmv(df)

    if rmv_value <= RMV_THRESHOLD:  # Filter stocks based on RMV condition
        results.append({"Ticker": ticker, "RMV": rmv_value})
    
    progress_bar.progress((i + 1) / total_stocks)
    time.sleep(0.5)  # To avoid rate limits

# Convert results to DataFrame and Display
final_df = pd.DataFrame(results)
if not final_df.empty:
    st.write("✅ **Filtered Stocks Based on RMV:**")
    st.dataframe(final_df)
    final_df.to_csv("filtered_stocks.csv", index=False)
    st.success("✅ Results saved to `filtered_stocks.csv`.")
else:
    st.warning("⚠️ No stocks met the RMV filter criteria.")

st.success("🚀 RMV Scanner Completed!")

