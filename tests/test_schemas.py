import pytest
from datetime import datetime, timezone
from src.schemas import Document, Signal


def test_document_schema_validation():
    doc = Document(
        doc_id="test_hash_123",
        source="newsapi",
        source_type="news",
        published_at=datetime.now(timezone.utc),
        text="Apple reports record quarterly revenue of $90B.",
        raw_entity_hints=["AAPL"]
    )
    assert doc.doc_id == "test_hash_123"
    assert doc.source_type == "news"
    assert "AAPL" in doc.raw_entity_hints


def test_signal_schema_mandated_fields():
    sig = Signal(
        signal_id="sig_001",
        doc_id="doc_001",
        timestamp=datetime.now(timezone.utc),
        source="financial_news",
        source_type="news",
        entity_type="company",
        ticker="AAPL",
        company="Apple Inc.",
        sentiment_score=0.85,
        sentiment_label="positive",
        event_type="Earnings/Guidance",
        event_confidence=0.92,
        impact_score=8,
        impact_confidence=0.88,
        text_excerpt="Apple beats quarterly earnings expectations.",
        model_version="distilroberta-v0.1.0"
    )
    assert sig.sentiment_score == 0.85
    assert sig.event_type == "Earnings/Guidance"
    assert sig.impact_score == 8
    assert 1 <= sig.impact_score <= 10
    assert -1.0 <= sig.sentiment_score <= 1.0


def test_signal_schema_invalid_ranges():
    with pytest.raises(Exception):
        Signal(
            signal_id="sig_bad",
            doc_id="doc_bad",
            timestamp=datetime.now(timezone.utc),
            source="test",
            source_type="news",
            entity_type="company",
            sentiment_score=2.5,  # Out of range [-1, 1]
            sentiment_label="positive",
            event_type="Geopolitical",
            event_confidence=0.5,
            impact_score=11,  # Out of range [1, 10]
            impact_confidence=0.5,
            text_excerpt="Bad ranges test",
            model_version="test"
        )
