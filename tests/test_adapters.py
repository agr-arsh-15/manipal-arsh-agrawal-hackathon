import pytest
from src.ingestion.adapters import PhraseBankAdapter, StockTweetsAdapter


def test_phrasebank_adapter():
    adapter = PhraseBankAdapter()
    docs = adapter.load_documents(limit=10)
    assert len(docs) == 10
    first = docs[0]
    assert first.source == "financial_phrasebank"
    assert first.source_type == "news"
    assert "ground_truth_sentiment" in first.metadata
    assert first.metadata["ground_truth_sentiment"] in [-1.0, 0.0, 1.0]


def test_stock_tweets_adapter():
    adapter = StockTweetsAdapter()
    docs = adapter.load_documents(limit=10)
    assert len(docs) == 10
    first = docs[0]
    assert first.source == "stock_tweets"
    assert first.source_type == "social"
    assert len(first.raw_entity_hints) > 0
