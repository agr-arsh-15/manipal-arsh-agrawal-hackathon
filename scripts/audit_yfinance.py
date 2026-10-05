import yfinance as yf
import pandas as pd
import sys

def probe_yfinance():
    tickers = ["SPY", "^VIX", "AAPL", "MSFT", "NVDA"]
    print(f"Testing yfinance historical cache for: {tickers}...")
    try:
        data = yf.download(tickers, start="2020-01-01", end="2020-01-10", progress=False)
        print("Data columns structure:")
        print(data.head(2))
        print("\nNull counts:")
        print(data['Close'].isna().sum())
        print("\nDownload succeeded. Shape:", data.shape)
    except Exception as e:
        print("yfinance probe error:", e)

if __name__ == "__main__":
    probe_yfinance()
