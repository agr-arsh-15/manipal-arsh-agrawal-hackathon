import yaml
import time
import os
import yfinance as yf
import pandas as pd

def cache_universe():
    with open("config/universe.yaml", "r") as f:
        data = yaml.safe_load(f)
    tickers = [entry["ticker"] for entry in data["universe"]]
    tickers = ["SPY", "^VIX"] + tickers
    
    out_dir = "data/prices"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Caching price history for {len(tickers)} symbols (2018-2024)...")
    for t in tickers:
        out_path = os.path.join(out_dir, f"{t}.csv")
        if os.path.exists(out_path):
            print(f"[{t}] already cached.")
            continue
        print(f"Fetching [{t}]...")
        time.sleep(1.0)
        df = yf.download(t, start="2018-01-01", end="2024-01-01", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(out_path)
            print(f"Saved [{t}] ({len(df)} rows)")
        else:
            print(f"Failed [{t}]")

if __name__ == "__main__":
    cache_universe()
