"""
Risk Engine API.

    python -m src.api.main            # http://localhost:8000/docs
    uvicorn src.api.main:app --reload
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.engine.pipeline import RiskEngine
from src.engine.store import SignalStore
from src.ingestion.gdelt import GdeltAdapter
from src.modules.rebalancer import RebalancerConfig, SentimentRebalancer, load_universe
from src.modules.stress import StressTestEngine
from src.schemas import Signal

SIGNALS_PATH = os.environ.get("RISK_SIGNALS_PATH", "data/samples/signals.jsonl")
BACKEND = os.environ.get("RISK_ENGINE_BACKEND", "auto")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = RiskEngine(backend=BACKEND)
    state["store"] = SignalStore(SIGNALS_PATH)
    yield
    state.clear()


app = FastAPI(
    title="AI/NLP Financial Risk Engine",
    description="Turns unstructured news, social and event-feed text into structured risk signals "
                "(sentiment_score, event_type, impact_score) for downstream portfolio modules.",
    version="1.0.0",
    lifespan=lifespan,
)


class AnalyzeItem(BaseModel):
    text: str = Field(min_length=3, max_length=5000)
    source: str = "api"
    source_type: Literal["news", "social", "event_feed"] = "news"
    published_at: Optional[datetime] = None


class AnalyzeRequest(BaseModel):
    items: List[AnalyzeItem] = Field(min_length=1, max_length=256)
    persist: bool = Field(default=True, description="Append resulting signals to signals.jsonl")


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_version: str
    signals_stored: int


@app.get("/health", response_model=HealthResponse)
def health():
    engine: RiskEngine = state["engine"]
    return HealthResponse(status="ok", backend=engine.backend, model_version=engine.model_version,
                          signals_stored=state["store"].count())


@app.post("/analyze", response_model=List[Signal])
def analyze(req: AnalyzeRequest):
    engine: RiskEngine = state["engine"]
    signals: List[Signal] = []
    for item in req.items:
        signals += engine.analyze_texts([item.text], source=item.source, source_type=item.source_type,
                                        published_at=item.published_at)
    if req.persist:
        state["store"].append(signals)
    return signals


@app.get("/signals", response_model=List[Signal])
def list_signals(
    event_type: Optional[str] = None,
    min_impact: Optional[int] = Query(default=None, ge=1, le=10),
    since: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=5000),
):
    return state["store"].read(event_type=event_type, min_impact=min_impact, since=since, limit=limit)


@app.get("/signals/{ticker}", response_model=List[Signal])
def signals_for_ticker(ticker: str, limit: int = Query(default=100, ge=1, le=5000)):
    if ticker.upper() not in state["engine"].linker.ticker_to_entity:
        raise HTTPException(status_code=404, detail=f"{ticker} is not in the monitored universe")
    return state["store"].read(ticker=ticker, limit=limit)


@app.post("/ingest/gdelt", response_model=List[Signal])
def ingest_gdelt(max_records: int = Query(default=50, ge=1, le=250), persist: bool = True):
    adapter = GdeltAdapter()
    docs = adapter.load_documents(max_records=max_records)
    signals = state["engine"].analyze_documents(docs)
    if persist and adapter.last_fetch_was_live:
        state["store"].append(signals)
    return signals


class StressRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Headline to analyse; overrides event_type/impact")
    event_type: Optional[str] = None
    impact_score: Optional[int] = Field(default=None, ge=1, le=10)
    force: bool = Field(default=False, description="Run even if the signal does not meet the trigger rule")


@app.post("/modules/stress-test")
def stress_test(req: StressRequest):
    """Module B: run the event-driven stress test for a headline or an explicit (event_type, impact)."""
    stress = state.setdefault("stress", StressTestEngine())
    if req.text:
        signal = state["engine"].analyze_texts([req.text], source="api")[0]
        result = stress.run_for_signal(signal, force=req.force)
        if result is None:
            return {"triggered": False, "signal": signal, "trigger_rule": stress.trigger}
    elif req.event_type and req.impact_score:
        result = stress.run(stress.shock_for(req.event_type, req.impact_score))
    else:
        raise HTTPException(status_code=422, detail="Provide text, or event_type and impact_score")
    result.pop("positions", None)
    return {"triggered": True, **result}


@app.get("/modules/rebalancer/weights")
def rebalancer_weights(halflife_days: float = 5.0, tilt_lambda: float = 3.0):
    """Module A: current sentiment-tilted target weights from the most recent stored signals."""
    tickers = load_universe()
    model = SentimentRebalancer(tickers, RebalancerConfig(halflife_days=halflife_days, tilt_lambda=tilt_lambda))
    latest = {}
    for s in state["store"].read():
        if s.ticker in tickers and s.ticker not in latest:
            latest[s.ticker] = s.sentiment_score
    sentiment = np.array([latest.get(t, 0.0) for t in tickers])
    weights = model.target_weights(sentiment)
    return [{"ticker": t, "latest_sentiment": round(float(s), 4), "target_weight": round(float(w), 4)}
            for t, s, w in zip(tickers, sentiment, weights)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
