import streamlit as st
import pandas as pd
import numpy as np
from polygon import RESTClient
import time

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

# Helper Functions
def calculate_rmv(df, lookback=50):
    """Calculate Relative Measured Volatility (RMV)"""
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

# Streamlit UI Components
uploaded_file = st.file_uploader("Upload TradingView Stock List (CSV)", type="csv")
account_balance = st.number_input("Account Balance ($)", min_value=1.0, value=10000.0)

if uploaded_file and st.button("Run Scanner"):
    try:
        # ✅ Fixed: Use 'on_bad_lines' instead of deprecated 'error_bad_lines'
        tv_df = pd.read_csv(uploaded_file, on_bad_lines="skip")
        if "Ticker" not in tv_df.columns:
            raise ValueError("CSV file must have a 'Ticker' column.")
        tickers = tv_df['Ticker'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    client = RESTClient(api_key=api_key)

    for i, ticker in enumerate(tickers):
        try:
            progress = (i+1)/len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"Processing {ticker} ({i+1}/{len(tickers)})")

            resp = client.get_aggs(ticker, 1, "day", "2023-01-01", "2024-01-01", limit=50000)

            # ✅ Debugging: Print API response to check format
            st.write(f"API Response for {ticker}: ", resp)

            # ✅ Fix: Convert `Agg` objects to dictionary format before creating DataFrame
            if isinstance(resp, list):
                formatted_data = [{
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                    "vwap": agg.vwap,
                    "timestamp": agg.timestamp
                } for agg in resp]

                df = pd.DataFrame(formatted_data)
            else:
                st.warning(f"Skipping {ticker}: No valid data received from API.")
                continue

            if df.empty:
                st.warning(f"Skipping {ticker}: No trading data available.")
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            df = calculate_rmv(df)
            if df is None or df.empty:
                continue

            latest = df.iloc[-1]
            if latest['rmv'] <= 20:
                entry_price = latest['close']
                atr = latest['atr5']
                stop_loss = entry_price - (1.5 * atr)
                target_price = entry_price + (2 * (entry_price - stop_loss))

                risk_amount = 0.01 * account_balance
                risk_per_share = entry_price - stop_loss
                position_size = int(risk_amount / risk_per_share) if risk_per_share > 0.01 else 0

                results.append({
                    'Ticker': ticker,
                    'RMV': round(latest['rmv'], 2),
                    'Entry': round(entry_price, 2),
                    'Stop Loss': round(stop_loss, 2),
                    'Target': round(target_price, 2),
                    'Shares': position_size
                })

            time.sleep(12)  # ✅ API rate limit fix

        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

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
        st.warning("No qualifying stocks found with RMV ≤ 20")

    progress_bar.empty()
    status_text.empty()


