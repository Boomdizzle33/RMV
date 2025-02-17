import streamlit as st
import pandas as pd
import numpy as np
import time
from polygon import RESTClient

# ✅ Load API Key from Streamlit Secrets
try:
    POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
except KeyError:
    st.error("❌ Polygon API key not found! Make sure it's set in secrets.toml")
    st.stop()

# ✅ Initialize Polygon.io Client
client = RESTClient(POLYGON_API_KEY)

# ✅ Streamlit UI
st.title("📈 RMV Stock Scanner")

# ✅ User Input for Account Size
account_size = st.number_input("Enter your account size ($)", min_value=1000, value=10000, step=500)
risk_per_trade = account_size * 0.01  # 1% risk per trade

# ✅ Upload CSV File
uploaded_file = st.file_uploader("Upload your TradingView CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Ensure "Ticker" Column Exists
    if "Ticker" not in df.columns:
        st.error("❌ CSV must contain a 'Ticker' column.")
        st.stop()

    st.success("✅ CSV Loaded Successfully!")
    st.dataframe(df)

    tickers = df["Ticker"].tolist()
    st.write(f"📊 Scanning {len(tickers)} stocks...")

    # ✅ Progress Bar
    progress_bar = st.progress(0)
    results = []

    # ✅ Fetch Data for Each Stock
    for i, ticker in enumerate(tickers):
        try:
            bars = client.get_aggs(ticker, 1, "day", "2024-01-01", "2024-02-15", limit=50)
            if not bars:
                continue

            # ✅ Convert Data to DataFrame
            prices = pd.DataFrame([{
                "date": bar["t"],
                "close": bar["c"],
                "high": bar["h"],
                "low": bar["l"],
                "volume": bar["v"]
            } for bar in bars])

            # ✅ Compute True Range and ATR
            prices["TrueRange"] = prices["high"] - prices["low"]
            prices["ATR"] = prices["TrueRange"].rolling(window=14).mean()
            prices["RMV"] = (prices["ATR"] / prices["close"]) * 100

            # ✅ Ensure RMV Data is Available
            if prices["RMV"].dropna().empty:
                st.warning(f"⚠️ No RMV data for {ticker}")
                continue

            latest_rmv = prices["RMV"].iloc[-1]

            # ✅ Apply VCP Filters
            if latest_rmv < 20:  # RMV below 20 = low volatility contraction
                last_close = prices["close"].iloc[-1]
                atr = prices["ATR"].iloc[-1]

                # ✅ Position Sizing: 1% Risk Per Trade
                stop_loss = atr * 1.5  # Example: Stop loss at 1.5x ATR
                position_size = risk_per_trade / stop_loss
                shares = int(position_size // last_close)  # Round to whole shares

                results.append({
                    "Ticker": ticker,
                    "RMV": round(latest_rmv, 2),
                    "Last Close": round(last_close, 2),
                    "ATR": round(atr, 2),
                    "Stop Loss": round(stop_loss, 2),
                    "Shares": shares
                })

            # ✅ Update Progress
            progress_bar.progress((i + 1) / len(tickers))
            time.sleep(0.5)  # ✅ Avoid hitting API rate limits

        except Exception as e:
            st.warning(f"⚠️ Error fetching data for {ticker}: {e}")

    # ✅ Display Results
    if results:
        results_df = pd.DataFrame(results)
        st.success("🎯 Stocks Matching RMV Criteria")
        st.dataframe(results_df)

        # ✅ Download Button
        csv_output = results_df.to_csv(index=False)
        st.download_button("📥 Download Results", csv_output, "RMV_Scanner.csv", "text/csv")
    else:
        st.warning("❌ No stocks matched the RMV criteria.")



