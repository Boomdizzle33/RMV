import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
from polygon import RESTClient

# Streamlit App Configuration
st.set_page_config(page_title="Swing Trade Scanner", layout="wide")
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

# ✅ Helper function to calculate RMV
def calculate_rmv(df, lookback=20):  # ✅ Reduced lookback period for better results
    try:
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr5'] = df['tr'].rolling(5).mean()
        df['atr10'] = df['tr'].rolling(10).mean()
        df['atr15'] = df['tr'].rolling(15).mean()

        df['avg_atr'] = (df['atr5'] + df['atr10'] + df['atr15']) / 3
        df['max_avg_atr'] = df['avg_atr'].rolling(lookback).max()
        df['rmv'] = (df['avg_atr'] / (df['max_avg_atr'] + 1e-9)) * 100

        return df.dropna()
    except Exception as e:
        st.error(f"Error calculating RMV: {str(e)}")
        return None

# ✅ Function to fetch stock data
def fetch_stock_data(ticker, results, client):
    try:
        resp = client.get_aggs(ticker, 1, "day", "2023-01-01", "2024-01-01", limit=50000)

        # ✅ Debugging: Print API response
        st.write(f"API Response for {ticker}: ", resp)

        if not isinstance(resp, dict) or "results" not in resp or not resp["results"]:
            st.warning(f"Skipping {ticker}: No valid data received from API.")
            return
        
        df = pd.DataFrame(resp["results"])
        if df.empty:
            st.warning(f"Skipping {ticker}: No trading data available.")
            return

        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

        # ✅ Debugging: Show raw DataFrame
        st.write(f"Data for {ticker} before RMV calculation:", df.head())

        df = calculate_rmv(df)
        if df is None or df.empty:
            return

        # ✅ Debugging: Show RMV values
        st.write(f"RMV values for {ticker}:", df[['timestamp', 'close', 'rmv']].tail(10))

        latest = df.iloc[-1]

        # ✅ Debugging: Show latest RMV before filtering
        st.write(f"Latest RMV for {ticker}: {latest['rmv']}")

        if latest['rmv'] <= 25:  # ✅ Loosened filter from ≤ 20 to ≤ 25
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
                'Shares': 0  # Position sizing logic can be added later
            })

    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")

# ✅ Streamlit UI Components
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

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    client = RESTClient(api_key=api_key)

    # ✅ Multi-threading for Faster API Calls
    threads = []
    for i, ticker in enumerate(tickers):
        t = threading.Thread(target=fetch_stock_data, args=(ticker, results, client))
        threads.append(t)
        t.start()

        progress = (i + 1) / len(tickers)
        progress_bar.progress(progress)
        status_text.text(f"Processing {ticker} ({i+1}/{len(tickers)})")

        time.sleep(1)  # ✅ Allows multi-threaded requests without exceeding rate limits

    for t in threads:
        t.join()

    # ✅ Display results
    if results:
        results_df = pd.DataFrame(results)
        st.subheader("Qualified Trade Setups")
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Export Trade List",
            data=csv,
            file_name="trade_setups.csv",
            mime="text/csv"
        )
    else:
        st.warning("No qualifying stocks found with RMV ≤ 25")  # ✅ Updated message

    progress_bar.empty()
    status_text.empty()



