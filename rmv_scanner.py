import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
from datetime import datetime, timedelta

# Load API key securely from Streamlit secrets
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]

# Function to fetch historical data from Polygon.io with proper error handling
def fetch_stock_data(ticker, timespan="day", limit=50):
    end_date = datetime.today().strftime("%Y-%m-%d")  # Get today's date
    start_date = (datetime.today() - timedelta(days=limit * 2)).strftime("%Y-%m-%d")  # Get enough past data

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
    params = {"apiKey": POLYGON_API_KEY, "limit": limit}
    
    session = requests.Session()  # Use session for efficiency
    try:
        response = session.get(url, params=params, timeout=5)  # Add timeout to prevent hanging requests
        response.raise_for_status()  # Raise error if request fails
        
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"]
        else:
            st.warning(f"No data returned for {ticker}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return []

# Function to compute RMV
def compute_rmv(data):
    if len(data) < 20:  # Ensure enough data points
        return 0
    closes = np.array([entry["c"] for entry in data])  # Extract close prices
    log_returns = np.diff(np.log(closes))  # Log returns
    if np.std(log_returns) == 0:
        return 0  # Prevent division by zero
    volatility = np.std(log_returns) * np.sqrt(252)  # Annualized volatility
    return round(volatility * 100, 2)

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

            # Update progress
            progress_bar.progress((index + 1) / total_stocks)

            # Prevent API rate limits
            time.sleep(1.2)

        # Convert results to DataFrame and display
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Download as CSV
        st.download_button(
            label="Download RMV Results",
            data=results_df.to_csv(index=False),
            file_name="rmv_results.csv",
            mime="text/csv"
        )

