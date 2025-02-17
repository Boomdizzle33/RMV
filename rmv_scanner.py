import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

# Load API Key securely from Streamlit secrets
API_KEY = st.secrets["polygon"]["api_key"]

# Streamlit UI
st.title("Pre-Market RMV Scanner")
st.sidebar.header("Upload Your Stock List")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Ticker' column", type=['csv'])

# User inputs account size
account_size = st.sidebar.number_input("Enter Account Principal ($)", min_value=1000, value=10000, step=100)

# RMV & Risk Management Parameters
ATR_PERIODS = (5, 10, 20)
LOOKBACK = 50
RISK_PERCENTAGE = 0.01  # 1% risk per trade
RISK_REWARD_RATIO = 2  # 2:1 risk-reward ratio
DATA_LOOKBACK_DAYS = 200  # Extended historical data for better swing trading analysis

# Function to fetch historical data for individual tickers
def fetch_polygon_data(ticker, retries=3, delay=2):
    """Fetches historical data from Polygon.io with retry handling & data validation."""
    for attempt in range(retries):
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{DATA_LOOKBACK_DAYS}/now?apiKey={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json().get("results", [])
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                    
                    # Ensure all necessary columns exist
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_columns):
                        return df
                    else:
                        st.warning(f"⚠️ Incomplete data for {ticker}. Skipping...")
                        return None
            st.warning(f"⚠️ No data for {ticker} (Attempt {attempt+1}/{retries}). Retrying...")
        except Exception as e:
            st.error(f"❌ API Error fetching {ticker}: {e}")
        
        time.sleep(delay)  # Wait before retrying
    return None  # Return None after max retries

# Function to calculate ATR
def calculate_atr(data, period):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift(1))
    low_close = np.abs(data['Low'] - data['Close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()

# Function to compute RMV
def compute_rmv(data):
    atr1 = calculate_atr(data, ATR_PERIODS[0])
    atr2 = calculate_atr(data, ATR_PERIODS[1])
    atr3 = calculate_atr(data, ATR_PERIODS[2])

    avg_atr = (atr1 + atr2 + atr3) / 3
    min_atr = avg_atr.rolling(window=LOOKBACK).min()
    max_atr = avg_atr.rolling(window=LOOKBACK).max()

    rmv = ((avg_atr - min_atr) / (max_atr - min_atr)) * 100
    rmv = rmv.fillna(0)  # Fill NaN with 0
    
    data['RMV'] = rmv
    return data

# Function to calculate stop loss, position sizing, and profit target
def calculate_risk(data, account_size):
    try:
        last_close = data['Close'].iloc[-1]
        atr_value = calculate_atr(data, ATR_PERIODS[1]).iloc[-1]  # Use middle ATR for risk sizing

        stop_loss = last_close - (1.5 * atr_value)
        risk_per_trade = account_size * RISK_PERCENTAGE
        position_size = risk_per_trade / (last_close - stop_loss)

        # Compute profit target based on risk-reward ratio
        profit_target = last_close + ((last_close - stop_loss) * RISK_REWARD_RATIO)

        return stop_loss, position_size, profit_target
    except Exception as e:
        st.error(f"❌ Risk Calculation Error: {e}")
        return None, None, None

# Process uploaded stock list
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    
    if 'Ticker' not in stock_list.columns:
        st.error("Uploaded CSV must contain a 'Ticker' column.")
    else:
        tickers = stock_list['Ticker'].tolist()
        results = []
        progress_bar = st.progress(0)

        for i, ticker in enumerate(tickers):
            st.write(f"Scanning {ticker} ({i+1}/{len(tickers)})...")
            stock_data = fetch_polygon_data(ticker)

            if stock_data is not None and len(stock_data) >= LOOKBACK:
                stock_data = compute_rmv(stock_data)
                last_rmv = stock_data['RMV'].iloc[-1]

                if last_rmv < 20:
                    stop_loss, position_size, profit_target = calculate_risk(stock_data, account_size)
                    if stop_loss and position_size:
                        results.append([ticker, stock_data['Close'].iloc[-1], last_rmv, stop_loss, profit_target, round(position_size)])

            progress_bar.progress((i+1) / len(tickers))
            time.sleep(0.5)  # Rate limit handling

        if results:
            final_df = pd.DataFrame(results, columns=["Ticker", "Close Price", "RMV", "Stop Loss", "Profit Target", "Position Size"])
            st.subheader("Stocks with RMV < 20")
            st.dataframe(final_df)

            # Export to TradingView format
            tradingview_export = final_df[['Ticker']]
            tradingview_export.to_csv("tradingview_watchlist.csv", index=False)

            st.success("Exported TradingView watchlist: tradingview_watchlist.csv")
        else:
            st.warning("No stocks found with RMV < 20.")
