import hashlib
import os
import re
import pandas as pd
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional, Iterable
from src.schemas import Document


# Legacy / share-class tickers that should resolve to the canonical universe ticker.
TICKER_REMAP = {"FB": "META", "GOOG": "GOOGL"}

_DATE_DMY = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def generate_doc_id(source: str, native_id: str) -> str:
    """Produces reproducible SHA-1 hash for document ID."""
    content = f"{source}:{native_id}".encode("utf-8")
    return hashlib.sha1(content).hexdigest()


def load_universe(universe_config_path: str = "config/universe.yaml") -> List[dict]:
    with open(universe_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("universe", [])


def build_name_to_ticker(universe: List[dict]) -> Dict[str, str]:
    """Lower-cased company name / alias -> canonical ticker."""
    mapping = {}
    for entry in universe:
        for alias in [entry["name"]] + entry.get("aliases", []):
            mapping[alias.strip().lower()] = entry["ticker"].upper()
    return mapping


class PhraseBankAdapter:
    """Ingests FinancialPhraseBank sentiment dataset (Malo et al., 50% agreement split)."""

    SENTIMENT_MAPPING = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(self, file_path: str = "data/raw/phrasebank/all-data.csv"):
        self.file_path = file_path

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, encoding="latin-1", header=None, names=["sentiment", "text"])

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        df = self.load_frame()
        if limit:
            df = df.iloc[:limit]

        documents = []
        for idx, row in df.iterrows():
            sentiment_str = str(row["sentiment"]).strip().lower()
            text = str(row["text"]).strip()
            documents.append(Document(
                doc_id=generate_doc_id("phrasebank", str(idx)),
                source="financial_phrasebank",
                source_type="news",
                # PhraseBank is undated; a fixed placeholder keeps it out of any chronological split.
                published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                time_precision="day",
                text=text,
                raw_entity_hints=[],
                metadata={
                    "ground_truth_sentiment": self.SENTIMENT_MAPPING.get(sentiment_str, 0.0),
                    "sentiment_label": sentiment_str,
                },
            ))
        return documents


class StockTweetsAdapter:
    """
    Ingests "Tweet Sentiment's Impact on Stock Returns" (Kaggle).

    The raw CSV contains multi-line tweets that shift columns for a subset of rows, the
    STOCK column holds a company *name* (e.g. "Apple"), and DATE is dd/mm/yyyy. Rows that
    fail structural validation are dropped rather than guessed.
    """

    def __init__(
        self,
        file_path: str = "data/raw/stock_tweets/reduced_dataset-release.csv",
        universe_config_path: str = "config/universe.yaml",
        universe_only: bool = True,
    ):
        self.file_path = file_path
        self.universe_only = universe_only
        self.name_to_ticker = build_name_to_ticker(load_universe(universe_config_path))

    def load_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path, low_memory=False)
        df = df[df["DATE"].astype(str).str.match(_DATE_DMY)].copy()
        df["published_at"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", utc=True)
        df["ticker"] = df["STOCK"].astype(str).str.strip().str.lower().map(self.name_to_ticker)
        for col in ["1_DAY_RETURN", "2_DAY_RETURN", "VOLATILITY_10D", "LSTM_POLARITY", "TEXTBLOB_POLARITY"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["TWEET"] = df["TWEET"].astype(str).str.strip()
        if self.universe_only:
            df = df[df["ticker"].notna()]
        return df.reset_index().rename(columns={"index": "native_idx"})

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        df = self.load_frame()
        if limit:
            df = df.iloc[:limit]

        documents = []
        for _, row in df.iterrows():
            ticker = row["ticker"] if isinstance(row["ticker"], str) else None
            documents.append(Document(
                doc_id=generate_doc_id("stock_tweets", str(row["native_idx"])),
                source="stock_tweets",
                source_type="social",
                published_at=row["published_at"].to_pydatetime(),
                time_precision="day",
                text=row["TWEET"],
                raw_entity_hints=[ticker] if ticker else [],
                metadata={
                    "company_name": row["STOCK"],
                    "return_1d": row["1_DAY_RETURN"],
                    "return_2d": row["2_DAY_RETURN"],
                    "volatility_10d": row["VOLATILITY_10D"],
                    "lstm_polarity": row["LSTM_POLARITY"],
                    "textblob_polarity": row["TEXTBLOB_POLARITY"],
                },
            ))
        return documents


class BenzingaAdapter:
    """
    Ingests the Benzinga "Massive Stock News" archive (raw_analyst_ratings.csv), which carries
    timezone-aware publication timestamps and a native ticker per headline.
    """

    def __init__(
        self,
        file_path: str = "data/raw/benzinga/raw_analyst_ratings.csv",
        universe_config_path: str = "config/universe.yaml",
        text_column: str = "headline",
        start_date: str = "2018-04-01",
    ):
        self.file_path = file_path
        self.feed = os.path.splitext(os.path.basename(file_path))[0]
        self.text_column = text_column
        self.start_date = pd.Timestamp(start_date, tz="UTC")
        self.universe_tickers = {e["ticker"].upper() for e in load_universe(universe_config_path)}

    def _iter_chunks(self, chunksize: int = 300_000) -> Iterable[pd.DataFrame]:
        cols = [self.text_column, "date", "stock"]
        for chunk in pd.read_csv(self.file_path, usecols=cols, chunksize=chunksize):
            chunk = chunk.dropna()
            chunk["stock"] = chunk["stock"].astype(str).str.upper().replace(TICKER_REMAP)
            yield chunk[chunk["stock"].isin(self.universe_tickers)]

    def load_frame(self) -> pd.DataFrame:
        df = pd.concat(list(self._iter_chunks()), ignore_index=False)
        df["published_at"] = pd.to_datetime(df["date"], utc=True, errors="coerce", format="mixed")
        df = df[df["published_at"] >= self.start_date]
        df = df.rename(columns={self.text_column: "text", "stock": "ticker"})
        df["text"] = df["text"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["text", "ticker"])
        df = df.reset_index().rename(columns={"index": "native_idx"})
        df["feed"] = self.feed
        return df

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        df = self.load_frame()
        if limit:
            df = df.iloc[:limit]
        return [
            Document(
                doc_id=generate_doc_id("benzinga", f"{self.feed}_{row['native_idx']}"),
                source="benzinga_news",
                source_type="news",
                published_at=row["published_at"].to_pydatetime(),
                time_precision="minute",
                text=row["text"],
                raw_entity_hints=[row["ticker"]],
                metadata={"feed": self.feed},
            )
            for _, row in df.iterrows()
        ]
