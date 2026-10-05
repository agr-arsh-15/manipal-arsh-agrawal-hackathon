import yfinance as yf
import pandas as pd
import time
import os

def download_ticker_history(ticker, start="2018-01-01", end="2024-01-01"):
    out_dir = "data/prices"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    if os.path.exists(out_path):
        print(f"[{ticker}] already cached at {out_path}.")
        return
    print(f"Fetching {ticker} from yfinance...")
    time.sleep(1)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if not df.empty:
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(out_path)
        print(f"Saved {ticker} ({len(df)} rows) to {out_path}")
    else:
        print(f"Warning: {ticker} returned empty dataset.")

if __name__ == "__main__":
    download_ticker_history("SPY")
    download_ticker_history("^VIX")
    download_ticker_history("AAPL")
