import streamlit as st
import pandas as pd
import numpy as np
import logging
import threading
import time
from polygon import RESTClient

# ✅ Enable Debug Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Streamlit App Configuration
st.set_page_config(page_title="RMV-Based Swing Trade Scanner", layout="wide")
st.title("RMV-Based Swing Trade Scanner")

# ✅ Securely Load API Key from Streamlit Secrets
try:
    api_key = st.secrets["polygon"]["api_key"]
    if not api_key:
        st.error("API Key is missing! Add it to Streamlit secrets.")
        st.stop()
except KeyError:
    st.error("Secrets file not found. Please add your API Key in `.streamlit/secrets.toml` on Streamlit Cloud.")
    st.stop()

# ✅ Create a Single RESTClient Instance (Fixes Connection Pool Issue)
client = RESTClient(api_key=api_key)

# ✅ Corrected RMV Calculation
def calculate_rmv(df, lookback=10):
    """Compute RMV with optimized scaling and EMA ATR."""
    try:
        df = df.copy()
        df.dropna(inplace=True)

        # ✅ Compute True Range (TR)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # ✅ Use EMA for ATR
        df['atr5'] = df['tr'].ewm(span=5, adjust=False).mean()
        df['atr10'] = df['tr'].ewm(span=10, adjust=False).mean()
        df['atr15'] = df['tr'].ewm(span=15, adjust=False).mean()

        # ✅ Compute Average ATR
        df['avg_atr'] = (df['atr5'] + df['atr10'] + df['atr15']) / 3

        # ✅ FIX: Adjust RMV Normalization for Better Contraction Detection
        df["rmv"] = (df["avg_atr"] - df["avg_atr"].rolling(lookback).min()) / (
            df["avg_atr"].rolling(lookback).max() - df["avg_atr"].rolling(lookback).min() + 1e-9
        ) * 100

        # ✅ FIX: Ensure RMV is in correct range (should not always be 100)
        df["rmv"] = df["rmv"].clip(0, 100)

        return df.dropna(subset=['rmv'])

    except Exception as e:
        logger.error(f"Error calculating RMV: {str(e)}")
        return None

# ✅ Fetch Stock Data with Rate Limit Handling
def fetch_stock_data(ticker, results, debug_logs):
    try:
        logger.debug(f"Fetching data for {ticker}")

        # ✅ Rate Limiting - Ensures we don’t exceed Polygon’s API limits
        time.sleep(0.8)  

        resp = client.get_aggs(ticker, 1, "day", "2023-01-01", "2024-01-01", limit=50000)

        if isinstance(resp, list):
            df = pd.DataFrame([vars(agg) for agg in resp])
        elif isinstance(resp, dict) and "results" in resp:
            df = pd.DataFrame(resp["results"])
        else:
            debug_logs.append(f"Skipping {ticker}: Unexpected API response format.")
            return
        
        if df.empty:
            debug_logs.append(f"Skipping {ticker}: No trading data available.")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

        df = calculate_rmv(df)
        if df is None or df.empty():
            return

        # ✅ FIX: Check RMV Trend Over Last 10 Days
        df["rmv_below_20"] = df["rmv"].rolling(window=10).apply(lambda x: any(x <= 20), raw=True)

        if df["rmv_below_20"].iloc[-1] == 1:
            latest = df.iloc[-1]
            entry_price = latest['close']
            atr = latest['atr5']
            stop_loss = entry_price - (1.5 * atr)
            target_price = entry_price + (2 * (entry_price - stop_loss))

            results.append({
                'Ticker': ticker,
                'RMV': round(latest['rmv'], 2),
                'Entry': round(entry_price, 2),
                'Stop Loss': round(stop_loss, 2),
                'Target': round(target_price, 2),
                'Shares': 0
            })

    except Exception as e:
        debug_logs.append(f"Error processing {ticker}: {str(e)}")

# ✅ Streamlit UI
uploaded_file = st.file_uploader("Upload TradingView Stock List (CSV)", type="csv")
account_balance = st.number_input("Account Balance ($)", min_value=1.0, value=10000.0)

if uploaded_file and st.button("Run Scanner"):
    try:
        tv_df = pd.read_csv(uploaded_file, on_bad_lines="skip")
        if "Ticker" not in tv_df.columns:
            raise ValueError("CSV file must have a 'Ticker' column.")
        tickers = tv_df['Ticker'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    results = []
    debug_logs = []

    threads = []
    for ticker in tickers:
        t = threading.Thread(target=fetch_stock_data, args=(ticker, results, debug_logs))
        threads.append(t)
        t.start()
        time.sleep(0.8)  # ✅ Prevents hitting API rate limits

    for t in threads:
        t.join()

    for log in debug_logs:
        st.warning(log)

    if results:
        st.subheader("Qualified Trade Setups")
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("No qualifying stocks found with RMV ≤ 20")


