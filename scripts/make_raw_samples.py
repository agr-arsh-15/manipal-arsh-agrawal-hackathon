"""
Writes small, format-preserving extracts of every raw source to data/raw_samples/ so the repo
shows exactly what the ingestion adapters consume without committing ~1.1 GB of Kaggle dumps.

    python -m scripts.make_raw_samples
"""
import os

import pandas as pd

OUT = "data/raw_samples"
N = 300
SEED = 7


def head_sample(path: str, out_name: str, nrows: int = 200_000, **read_kw):
    df = pd.read_csv(path, nrows=nrows, on_bad_lines="skip", low_memory=False, **read_kw)
    df.sample(min(N, len(df)), random_state=SEED).to_csv(f"{OUT}/{out_name}", index=False)
    print(f"{out_name}: {min(N, len(df))} rows from {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    head_sample("data/raw/phrasebank/all-data.csv", "phrasebank_sample.csv",
                header=None, names=["sentiment", "text"], encoding="latin-1")
    head_sample("data/raw/benzinga/raw_analyst_ratings.csv", "benzinga_analyst_ratings_sample.csv")
    head_sample("data/raw/benzinga/raw_partner_headlines.csv", "benzinga_partner_headlines_sample.csv")
    head_sample("data/raw/stock_tweets/reduced_dataset-release.csv", "stock_tweets_sample.csv")


if __name__ == "__main__":
    main()
