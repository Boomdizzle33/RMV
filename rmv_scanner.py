import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import logging
from datetime import datetime, timedelta
from queue import Queue
from polygon import RESTClient

# ✅ Configure logging with thread-safe queue
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Streamlit App Configuration
st.set_page_config(page_title="Swing Trade Scanner", layout="wide")
st.title("RMV-Based Swing Trade Scanner")

# ✅ Initialize thread-safe queue for logging
debug_queue = Queue()

# ✅ Securely Load API Key
try:
    api_key = st.secrets["polygon"]["api_key"]
except KeyError:
    st.error("Missing API Key! Add it in Streamlit Secrets.")
    st.stop()

# ✅ Improved RMV Calculation with Validation
def calculate_rmv(df, lookback=20):
    """Compute RMV with error handling and validation checks"""
    try:
        if df.empty or len(df) < lookback:
            return None
            
        df = df.copy()
        required_columns = {'open', 'high', 'low', 'close'}
        if not required_columns.issubset(df.columns):
            raise ValueError("Missing required price columns")

        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['prev_close']).abs(),
            (df['low'] - df['prev_close']).abs()
        ], axis=1).max(axis=1)

        # EMA-based ATR calculation
        periods = [5, 10, 15]
        for period in periods:
            df[f'atr{period}'] = df['tr'].ewm(span=period, adjust=False).mean()
        
        df['avg_atr'] = df[[f'atr{p}' for p in periods]].mean(axis=1)
        df['max_avg_atr'] = df['avg_atr'].rolling(lookback).max()
        df['rmv'] = (df['avg_atr'] / (df['max_avg_atr'] + 1e-9)) * 100

        return df.dropna(subset=['rmv']).tail(100)  # Return last 100 days

    except Exception as e:
        debug_queue.put(f"RMV Calculation Error: {str(e)}")
        return None

# ✅ Enhanced Data Fetching with Rate Limiting
def fetch_stock_data(ticker, results, client, account_balance):
    try:
        # Dynamic date range (last 365 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch data with error handling
        resp = client.get_aggs(
            ticker, 
            1, 
            "day", 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d"),
            limit=50000
        )

        # Convert response to DataFrame
        if not resp:
            debug_queue.put(f"No data for {ticker}")
            return
            
        data = []
        for agg in resp:
            if hasattr(agg, 'open') and hasattr(agg, 'close'):
                data.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
        
        if not data:
            debug_queue.put(f"Empty response for {ticker}")
            return
            
        df = pd.DataFrame(data)

        # Calculate RMV and trading parameters
        df = calculate_rmv(df)
        if df is None or df.empty:
            return

        latest = df.iloc[-1]
        if latest['rmv'] <= 25:
            # Position sizing calculations
            entry_price = latest['close']
            atr = latest['atr5']
            stop_loss = entry_price - (1.5 * atr)
            risk_per_share = entry_price - stop_loss
            
            # Risk 1% of account per trade
            risk_amount = account_balance * 0.01
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            
            results.append({
                'Ticker': ticker,
                'RMV': round(latest['rmv'], 2),
                'Entry': round(entry_price, 2),
                'Stop Loss': round(stop_loss, 2),
                'Target': round(entry_price + (2 * risk_per_share), 2),
                'Shares': shares,
                'Risk ($)': round(risk_per_share * shares, 2)
            })

    except Exception as e:
        debug_queue.put(f"Error processing {ticker}: {str(e)}")

# Streamlit UI Components
def main():
    uploaded_file = st.file_uploader("Upload Stock List (CSV)", type="csv")
    account_balance = st.number_input("Account Balance ($)", min_value=100.0, value=10000.0, step=1000.0)

    if uploaded_file and st.button("Run Scanner"):
        try:
            # Process CSV input
            tv_df = pd.read_csv(uploaded_file)
            if "Ticker" not in tv_df.columns:
                raise ValueError("CSV must contain 'Ticker' column")
                
            tickers = tv_df['Ticker'].dropna().unique().tolist()
            if not tickers:
                raise ValueError("No valid tickers found in CSV")

            st.info(f"Scanning {len(tickers)} tickers...")
            
            # Initialize shared resources
            results = []
            client = RESTClient(api_key=api_key)
            rate_limit_delay = 12  # Polygon free tier: 5 requests/minute

            # Start processing threads
            threads = []
            for ticker in tickers:
                t = threading.Thread(
                    target=fetch_stock_data,
                    args=(ticker, results, client, account_balance)
                )
                t.start()
                threads.append(t)
                time.sleep(rate_limit_delay)  # ✅ Enforce rate limiting

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Display debug logs
            st.subheader("Process Logs")
            while not debug_queue.empty():
                st.text(debug_queue.get())

            # Display results
            st.subheader("Trade Opportunities")
            if results:
                df = pd.DataFrame(results)
                df = df.sort_values('RMV').reset_index(drop=True)
                st.dataframe(df.style.highlight_max(subset=['Target'], color='lightgreen'))
                
                # Risk management summary
                total_risk = df['Risk ($)'].sum()
                st.metric("Total Portfolio Risk", f"${total_risk:,.2f} ({total_risk/account_balance*100:.1f}%)")
            else:
                st.warning("No qualifying trades found (RMV ≤ 25)")

        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

if __name__ == "__main__":
    main()

