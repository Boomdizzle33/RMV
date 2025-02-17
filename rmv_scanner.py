import pandas as pd
import requests
import numpy as np
import time
import os

# Load API key securely
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # Ensure it's set in your environment or Streamlit secrets

# Load pre-filtered list of stocks
CSV_FILE = "your_watchlist.csv"  # Make sure this file exists in your working directory

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Error: {CSV_FILE} not found. Make sure the file is in the correct location.")

df = pd.read_csv(CSV_FILE)

if "Ticker" not in df.columns:
    raise ValueError("Error: CSV file must contain a 'Ticker' column.")

tickers = df["Ticker"].dropna().tolist()

if not tickers:
    raise ValueError("Error: No tickers found in the CSV file.")

# Polygon.io API parameters
BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=120&apiKey=" + POLYGON_API_KEY

# Date range (modify if needed)
from datetime import datetime, timedelta

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')  # Get last 120 days of data

def fetch_stock_data(ticker):
    """Fetches historical stock data from Polygon.io"""
    url = BASE_URL.format(ticker=ticker, start_date=start_date, end_date=end_date)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Warning: API request failed for {ticker} (Status Code {response.status_code})")
        return None
    
    data = response.json()
    
    if "results" not in data or not isinstance(data["results"], list):
        print(f"Warning: No valid data received for {ticker}")
        return None
    
    return data["results"]

def calculate_atr(data, period=14):
    """Calculates the Average True Range (ATR) for a given stock"""
    if len(data) < period:
        print(f"Skipping ATR calculation: Not enough data ({len(data)} days provided)")
        return None

    highs = np.array([day["h"] for day in data])
    lows = np.array([day["l"] for day in data])
    closes = np.array([day["c"] for day in data])
    
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, shift=1))
    tr3 = np.abs(lows - np.roll(closes, shift=1))
    
    tr = np.maximum(np.maximum(tr1, tr2), tr3)[1:]  # Ignore first NaN row
    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    
    return atr[-1] if len(atr) > 0 else None  # Return latest ATR value

# Run RMV Calculation
results = []
for i, ticker in enumerate(tickers):
    print(f"Processing {ticker} ({i+1}/{len(tickers)})...")

    stock_data = fetch_stock_data(ticker)
    if stock_data is None:
        continue

    atr = calculate_atr(stock_data)
    
    if atr is None:
        print(f"Skipping {ticker}: ATR could not be calculated.")
        continue

    latest_close = stock_data[-1]["c"]
    
    if latest_close == 0:
        print(f"Skipping {ticker}: Invalid closing price (0).")
        continue

    rmv = (atr / latest_close) * 100
    
    if rmv == 0:
        print(f"Warning: RMV is zero for {ticker}. Investigate data.")

    results.append({"Ticker": ticker, "Latest_RMV": rmv})

    time.sleep(1)  # Avoid hitting rate limits

# Save results to CSV
output_df = pd.DataFrame(results)
output_file = "rmv_results.csv"
output_df.to_csv(output_file, index=False)

print(f"RMV calculation completed. Results saved to {output_file}.")

