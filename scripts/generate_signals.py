"""
Runs the risk engine over every dated document in the demo window and writes the signal
stream that Module A and Module B consume (data/samples/signals.jsonl).

    python -m scripts.generate_signals [--backend auto|transformer|baseline]
"""
import argparse
from collections import Counter

import pandas as pd
import yaml

from src.engine.pipeline import RiskEngine
from src.engine.store import SignalStore
from src.ingestion.adapters import BenzingaAdapter, StockTweetsAdapter
from src.ingestion.gdelt import GdeltAdapter


def collect_documents(cfg: dict):
    ds = cfg["dataset"]
    docs = []
    for feed in ds["benzinga_feeds"]:
        docs += BenzingaAdapter(feed["path"], start_date=ds["start_date"]).load_documents()
    tweets = StockTweetsAdapter(ds["tweets_path"]).load_documents()
    start = pd.Timestamp(ds["start_date"], tz="UTC")
    seen_text = set()
    for d in tweets:
        if d.published_at >= start and d.text not in seen_text:
            seen_text.add(d.text)
            docs.append(d)
    docs += GdeltAdapter().load_snapshot()
    return docs


def main(backend: str):
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    engine = RiskEngine(backend=backend)
    docs = collect_documents(cfg)
    print(f"Backend={engine.backend} ({engine.model_version}); documents={len(docs)}")
    signals = engine.analyze_documents(docs, batch_size=128)
    store = SignalStore(cfg["paths"]["signals_file"])
    n = store.overwrite(signals)
    print(f"Wrote {n} signals to {store.path}")
    print("By source:", dict(Counter(s.source for s in signals)))
    print("By event_type:", dict(Counter(s.event_type for s in signals).most_common()))
    print("Impact > 7:", sum(s.impact_score > 7 for s in signals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto")
    main(parser.parse_args().backend)
