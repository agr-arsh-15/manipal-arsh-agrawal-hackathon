import pytest
from src.ingestion.adapters import BenzingaAdapter, PhraseBankAdapter, StockTweetsAdapter


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
    # STOCK holds company names ("Apple"); they must resolve to universe tickers.
    assert first.raw_entity_hints[0] in adapter.name_to_ticker.values()
    assert all(d.published_at.year in (2017, 2018) for d in docs)


def test_benzinga_adapter_remaps_legacy_tickers():
    adapter = BenzingaAdapter(start_date="2020-01-01")
    df = adapter.load_frame()
    assert len(df) > 0
    assert not df["ticker"].isin(["FB", "GOOG"]).any()
    assert df["ticker"].isin(adapter.universe_tickers).all()
    assert (df["published_at"] >= adapter.start_date).all()
