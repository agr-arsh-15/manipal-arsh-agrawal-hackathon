import hashlib
import pandas as pd
from datetime import datetime, timezone
from typing import List, Generator, Optional
from src.schemas import Document


def generate_doc_id(source: str, native_id: str) -> str:
    """Produces reproducible SHA-1 hash for document ID."""
    content = f"{source}:{native_id}".encode("utf-8")
    return hashlib.sha1(content).hexdigest()


class PhraseBankAdapter:
    """Ingests FinancialPhraseBank sentiment dataset."""

    SENTIMENT_MAPPING = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0
    }

    def __init__(self, file_path: str = "data/raw/phrasebank/all-data.csv"):
        self.file_path = file_path

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        df = pd.read_csv(self.file_path, encoding="latin-1", header=None, names=["sentiment", "text"])
        if limit:
            df = df.iloc[:limit]

        documents = []
        for idx, row in df.iterrows():
            sentiment_str = str(row["sentiment"]).strip().lower()
            text = str(row["text"]).strip()
            num_score = self.SENTIMENT_MAPPING.get(sentiment_str, 0.0)

            doc = Document(
                doc_id=generate_doc_id("phrasebank", str(idx)),
                source="financial_phrasebank",
                source_type="news",
                published_at=datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                time_precision="day",
                text=text,
                raw_entity_hints=[],
                metadata={
                    "ground_truth_sentiment": num_score,
                    "sentiment_label": sentiment_str
                }
            )
            documents.append(doc)
        return documents


class StockTweetsAdapter:
    """Ingests Stock Tweets dataset with return horizons and ticker mentions."""

    def __init__(self, file_path: str = "data/raw/stock_tweets/reduced_dataset-release.csv"):
        self.file_path = file_path

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        df = pd.read_csv(self.file_path)
        if limit:
            df = df.iloc[:limit]

        documents = []
        for idx, row in df.iterrows():
            ticker = str(row.get("STOCK", "")).strip().upper()
            date_str = str(row.get("DATE", "")).strip()
            text = str(row.get("TWEET", "")).strip()

            try:
                published_at = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                published_at = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

            doc = Document(
                doc_id=generate_doc_id("stock_tweets", f"{ticker}_{date_str}_{idx}"),
                source="stock_tweets",
                source_type="social",
                published_at=published_at,
                time_precision="day",
                text=text,
                raw_entity_hints=[ticker] if ticker else [],
                metadata={
                    "ticker": ticker,
                    "return_1d": row.get("1_DAY_RETURN"),
                    "return_2d": row.get("2_DAY_RETURN"),
                    "return_7d": row.get("7_DAY_RETURN"),
                    "volatility_10d": row.get("VOLATILITY_10D"),
                    "lstm_polarity": row.get("LSTM_POLARITY"),
                    "textblob_polarity": row.get("TEXTBLOB_POLARITY")
                }
            )
            documents.append(doc)
        return documents
