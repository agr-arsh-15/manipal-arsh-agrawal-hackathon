from datetime import datetime, timezone

import pytest

from src.engine.pipeline import RiskEngine
from src.engine.store import SignalStore
from src.schemas import Document


@pytest.fixture(scope="module")
def engine():
    return RiskEngine(backend="baseline")


def test_company_headline_produces_linked_signal(engine):
    signals = engine.analyze_texts(["Apple shares jump after iPhone revenue beats analyst estimates"])
    assert len(signals) == 1
    s = signals[0]
    assert s.ticker == "AAPL" and s.entity_type == "company"
    assert -1.0 <= s.sentiment_score <= 1.0
    assert 1 <= s.impact_score <= 10
    assert s.event_type
    assert 0.0 <= s.event_confidence <= 1.0 and 0.0 <= s.impact_confidence <= 1.0


def test_macro_headline_without_ticker_is_event_signal(engine):
    s = engine.analyze_texts(["Central bank raises interest rates to fight inflation"])[0]
    assert s.ticker is None and s.entity_type == "event" and s.event_id


def test_multi_ticker_document_fans_out(engine):
    doc = Document(doc_id="d1", source="test", source_type="social",
                   published_at=datetime(2020, 3, 1, tzinfo=timezone.utc),
                   text="$MSFT and $NVDA rally as cloud stocks rebound", raw_entity_hints=[])
    signals = engine.analyze_documents([doc])
    assert {s.ticker for s in signals} == {"MSFT", "NVDA"}
    assert all(s.timestamp == doc.published_at for s in signals)


def test_store_roundtrip_and_filters(engine, tmp_path):
    store = SignalStore(str(tmp_path / "signals.jsonl"))
    signals = engine.analyze_texts(["Apple shares rise as it launches new iPhone",
                                    "Exxon Mobil faces lawsuit over spill"])
    assert store.append(signals) == len(signals)
    assert store.count() == len(signals)
    assert [s.ticker for s in store.read(ticker="aapl")] == ["AAPL"]
    assert store.read(min_impact=11) == []
