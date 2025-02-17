import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
from datetime import datetime, timedelta

# Load API key securely from Streamlit secrets
try:
    POLYGON_API_KEY = st.secrets["polygon"]["api_key"]  # Matches your secrets.toml structure
except KeyError:
    st.error("API Key not found! Ensure it's stored correctly in secrets.toml under [polygon]")

# Function to fetch stock data with retry logic and exponential backoff
def fetch_stock_data(ticker, timespan="day", limit=50, max_retries=3):
    end_date = datetime.today().strftime("%Y-%m-%d")  
    start_date = (datetime.today() - timedelta(days=limit * 2)).strftime("%Y-%m-%d")  
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
    
    params = {"apiKey": POLYGON_API_KEY, "limit": limit}
    session = requests.Session()

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=10)  # Increased timeout
            response.raise_for_status()  
            data = response.json()
            
            if "results" in data and data["results"]:
                return data["results"]
            else:
                st.warning(f"No data found for {ticker}")
                return []
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching {ticker}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return []  # Return empty list if all retries fail

# Function to compute RMV
def compute_rmv(data):
    if len(data) < 20:  
        return 0  # Not enough data for calculation
    closes = np.array([entry["c"] for entry in data])  
    log_returns = np.diff(np.log(closes))  
    return round(np.std(log_returns) * np.sqrt(252) * 100, 2) if np.std(log_returns) != 0 else 0

# Streamlit UI
st.title("RMV Stock Scanner")
uploaded_file = st.file_uploader("Upload your stock list CSV (must contain 'Ticker' column)", type=["csv"])

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

            time.sleep(0.6)  # Optimized for Polygon.io $29 plan rate limits

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        st.download_button(
            label="Download RMV Results",
            data=results_df.to_csv(index=False),
            file_name="rmv_results.csv",
            mime="text/csv"
        )


