import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from polygon import RESTClient

# Load API Key from Streamlit Secrets
API_KEY = st.secrets["polygon"]["api_key"]

# Streamlit UI
st.set_page_config(page_title="RMV Swing Trade Scanner", layout="wide")
st.title("📊 RMV-Based Swing Trade Scanner")

# Upload Stock List
uploaded_file = st.file_uploader("📂 Upload TradingView Stock List (CSV)", type="csv")
account_balance = st.number_input("💰 Account Balance ($)", min_value=1.0, value=10000.0)

# Constants
LOOKBACK_PERIOD = 50  # For RMV Calculation
ATR_PERIODS = [5, 10, 15]  # ATR Lookbacks
RISK_PERCENTAGE = 0.01  # 1% Risk Per Trade

# Function to Calculate RMV
def calculate_rmv(df):
    try:
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        ])

        # Calculate ATR for multiple periods
        for period in ATR_PERIODS:
            df[f'atr{period}'] = df['tr'].rolling(period).mean()

        # Compute RMV
        df['avg_atr'] = df[[f'atr{p}' for p in ATR_PERIODS]].mean(axis=1)
        df['max_avg_atr'] = df['avg_atr'].rolling(LOOKBACK_PERIOD).max()
        df['rmv'] = (df['avg_atr'] / df['max_avg_atr']) * 100

        return df.dropna()  # Remove NaN values
    except Exception as e:
        st.error(f"⚠️ RMV Calculation Error: {e}")
        return None

# Function to Fetch Data from Polygon.io
def fetch_stock_data(ticker):
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2023-01-01/2024-01-01?limit=50000&apiKey={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if 'results' not in data or not data['results']:
            st.warning(f"⚠️ Skipping {ticker}: No trading data available.")
            return None

        # Convert API response to DataFrame
        df = pd.DataFrame(data['results'])
        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')

        return df
    except Exception as e:
        st.error(f"⚠️ API Error fetching {ticker}: {e}")
        return None

# Main Scanner Execution
if uploaded_file and st.button("🚀 Run Scanner"):
    tv_df = pd.read_csv(uploaded_file)
    tickers = tv_df['Ticker'].unique().tolist()
    
    results = []
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        progress = (i + 1) / len(tickers)
        progress_bar.progress(progress)
        
        df = fetch_stock_data(ticker)
        if df is None:
            continue

        df = calculate_rmv(df)
        if df is None or df.empty:
            continue

        latest = df.iloc[-1]
        if latest['rmv'] <= 20:
            entry_price = latest['close']
            atr = latest['atr5']
            stop_loss = entry_price - (1.5 * atr)
            target_price = entry_price + (2 * (entry_price - stop_loss))

            risk_amount = account_balance * RISK_PERCENTAGE
            risk_per_share = entry_price - stop_loss
            position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

            results.append({
                'Ticker': ticker,
                'RMV': round(latest['rmv'], 2),
                'Entry': round(entry_price, 2),
                'Stop Loss': round(stop_loss, 2),
                'Target': round(target_price, 2),
                'Shares': position_size
            })

        time.sleep(1)  # API rate limit handling

    progress_bar.empty()

    if results:
        results_df = pd.DataFrame(results)
        st.subheader("✅ Qualified Trade Setups")
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button(label="📥 Export Trade List", data=csv, file_name="trade_setups.csv", mime="text/csv")
    else:
        st.warning("⚠️ No qualifying stocks found with RMV ≤ 20")





