import hashlib
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.eval.metrics import sentiment_to_label
from src.ingestion.adapters import generate_doc_id
from src.linking.matcher import EntityLinker
from src.models.baselines import BaselineModelSuite
from src.schemas import Document, Signal

TRANSFORMER_DIR = "models/risk_engine"
BASELINE_DIR = "models/baseline"
MAX_TICKERS_PER_DOC = 3
# Residual spread of the TF-IDF impact regressor on held-out data (1-10 scale); used to give
# baseline predictions an honest, constant impact confidence.
BASELINE_IMPACT_SIGMA = 2.9


class _BaselineBackend:
    def __init__(self, model_dir: str):
        self.suite = BaselineModelSuite()
        self.suite.load(model_dir)
        self.version = "baseline-tfidf-linear"

    def predict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        X_e = self.suite.tfidf_event.transform(texts)
        probs = self.suite.model_event.predict_proba(X_e)
        classes = self.suite.model_event.classes_
        sigma = np.full(len(texts), BASELINE_IMPACT_SIGMA)
        return {
            "sentiment": self.suite.predict_sentiment(texts),
            "event_type": classes[probs.argmax(1)],
            "event_confidence": probs.max(1),
            "impact_raw": self.suite.predict_impact(texts),
            "impact_sigma": sigma,
            "impact_confidence": np.array([math.erf(1.5 / (s * math.sqrt(2))) for s in sigma]),
        }


class RiskEngine:
    """
    End-to-end AI/NLP risk engine: Document -> entity linking -> multi-task model -> Signal.

    Backend selection: the fine-tuned transformer in models/risk_engine is used when present;
    otherwise the committed TF-IDF baselines keep the engine fully functional.
    """

    def __init__(
        self,
        backend: str = "auto",
        transformer_dir: str = TRANSFORMER_DIR,
        baseline_dir: str = BASELINE_DIR,
        device: str = "auto",
    ):
        self.linker = EntityLinker("config/universe.yaml")
        use_transformer = backend == "transformer" or (
            backend == "auto" and os.path.exists(os.path.join(transformer_dir, "heads.pt"))
        )
        if use_transformer:
            from src.models.multitask import RiskModelPredictor
            self.model = RiskModelPredictor(transformer_dir, device=device)
            self.backend = "transformer"
        else:
            self.model = _BaselineBackend(baseline_dir)
            self.backend = "baseline"
        self.model_version = self.model.version

    def predict_raw(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        return self.model.predict(list(texts))

    def analyze_documents(self, docs: Sequence[Document], batch_size: int = 64) -> List[Signal]:
        signals: List[Signal] = []
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            preds = self.predict_raw([d.text for d in batch])
            labels = sentiment_to_label(preds["sentiment"])
            for i, doc in enumerate(batch):
                links = self.linker.link(doc.text, doc.raw_entity_hints)[:MAX_TICKERS_PER_DOC]
                targets = links or [(None, None, None)]
                for ticker, company, _ in targets:
                    signals.append(Signal(
                        signal_id=hashlib.sha1(f"{doc.doc_id}:{ticker}".encode()).hexdigest()[:16],
                        doc_id=doc.doc_id,
                        timestamp=doc.published_at,
                        source=doc.source,
                        source_type=doc.source_type,
                        entity_type="company" if ticker else "event",
                        ticker=ticker,
                        company=company,
                        event_id=None if ticker else doc.doc_id[:16],
                        sentiment_score=round(float(np.clip(preds["sentiment"][i], -1, 1)), 4),
                        sentiment_label=labels[i],
                        event_type=str(preds["event_type"][i]),
                        event_confidence=round(float(preds["event_confidence"][i]), 4),
                        impact_score=int(np.clip(round(float(preds["impact_raw"][i])), 1, 10)),
                        impact_confidence=round(float(preds["impact_confidence"][i]), 4),
                        text_excerpt=doc.text[:200],
                        model_version=self.model_version,
                    ))
        return signals

    def analyze_texts(
        self,
        texts: Sequence[str],
        source: str = "api",
        source_type: str = "news",
        published_at: Optional[datetime] = None,
    ) -> List[Signal]:
        ts = published_at or datetime.now(timezone.utc)
        docs = [
            Document(doc_id=generate_doc_id(source, f"{t}|{ts.isoformat()}"), source=source,
                     source_type=source_type, published_at=ts, text=t)
            for t in texts
        ]
        return self.analyze_documents(docs)

    def benchmark_latency(self, text: str, runs: int = 50) -> Dict[str, float]:
        self.predict_raw([text])
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict_raw([text])
            times.append((time.perf_counter() - t0) * 1000)
        return {"p50_ms": round(float(np.percentile(times, 50)), 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2)}
