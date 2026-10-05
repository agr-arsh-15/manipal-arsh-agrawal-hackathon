from datetime import datetime
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Unified raw document schema produced by all data ingestion adapters.
    """
    doc_id: str = Field(description="Unique hash identifier (SHA-1) of source and native ID")
    source: str = Field(description="Data origin identifier, e.g., 'financial_phrasebank', 'stock_tweets', 'gdelt', 'newsapi'")
    source_type: Literal["news", "social", "event_feed"] = Field(description="Channel modality")
    published_at: datetime = Field(description="UTC publication timestamp")
    time_precision: Literal["minute", "hour", "day"] = Field(default="minute", description="Timestamp granularity for leakage control")
    text: str = Field(description="Raw body or headline text")
    url: Optional[str] = Field(default=None, description="Original article/source link if available")
    raw_entity_hints: Optional[List[str]] = Field(default_factory=list, description="Native tags or ticker mentions provided by source")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Supplementary source-specific fields")


class Signal(BaseModel):
    """
    Unified downstream financial risk signal schema.
    Strictly guarantees presence of the 3 hackathon-mandated fields:
    - sentiment_score
    - event_type
    - impact_score
    """
    signal_id: str = Field(description="Unique signal identifier UUID/hash")
    doc_id: str = Field(description="Originating Document identifier")
    timestamp: datetime = Field(description="UTC execution or document publication timestamp")
    source: str = Field(description="Source identifier")
    source_type: Literal["news", "social", "event_feed"] = Field(description="Modality")
    
    # Entity Resolution
    entity_type: Literal["company", "event"] = Field(description="'company' for ticker-resolved events, 'event' for macro")
    ticker: Optional[str] = Field(default=None, description="Standard ticker symbol (e.g., AAPL) or None if macro")
    company: Optional[str] = Field(default=None, description="Resolved company legal name")
    event_id: Optional[str] = Field(default=None, description="Global event cluster ID if macro/GDELT")
    
    # Mandated Field 1: Sentiment Score [-1.0, 1.0]
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Numerical sentiment polarity score")
    sentiment_label: Literal["positive", "neutral", "negative"] = Field(description="Discrete categorical sentiment classification")
    
    # Mandated Field 2: Event Type
    event_type: str = Field(description="Categorical event label matching taxonomy")
    event_confidence: float = Field(ge=0.0, le=1.0, description="Softmax confidence or rule certainty for event classification")
    
    # Mandated Field 3: Impact Score [1, 10]
    impact_score: int = Field(ge=1, le=10, description="Calibrated market-impact severity scale (1 to 10)")
    impact_confidence: float = Field(ge=0.0, le=1.0, description="Confidence interval or empirical certainty score")
    
    # Provenance and Traceability
    text_excerpt: str = Field(description="Brief text excerpt (<=200 chars) for auditability")
    model_version: str = Field(description="Model identifier and checkpoint version")
