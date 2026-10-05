import os
import yaml
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.schemas import Document, Signal
from src.linking.matcher import EntityLinker
from src.labeling.rules import EventTaxonomyLabeler
from src.labeling.impact import MarketImpactEngine

def build_dataset():
    print("Building unified labeled dataset across news and social modalities...")
    linker = EntityLinker("config/universe.yaml")
    labeler = EventTaxonomyLabeler("config/taxonomy.yaml")
    impact_engine = MarketImpactEngine("data/prices", "config/engine.yaml")
    
    # 1. Ingest PhraseBank (Gold Sentiment)
    phrasebank_path = "data/raw/phrasebank/all-data.csv"
    df_pb = pd.read_csv(phrasebank_path, encoding="latin-1", header=None, names=["sentiment", "text"])
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    
    records = []
    
    print(f"Processing FinancialPhraseBank ({len(df_pb)} rows)...")
    for idx, row in df_pb.iterrows():
        text = str(row["text"]).strip()
        sent_label = str(row["sentiment"]).strip().lower()
        sent_score = sentiment_map.get(sent_label, 0.0)
        
        event_type, event_conf = labeler.label(text)
        links = linker.link(text)
        ticker = links[0][0] if links else None
        company = links[0][1] if links else None
        entity_type = "company" if ticker else "event"
        
        # Phrasebank has no date, so impact is unobserved / masked
        records.append({
            "doc_id": f"phrasebank_{idx}",
            "source": "financial_phrasebank",
            "source_type": "news",
            "published_at": "2020-01-01T00:00:00Z",
            "text": text,
            "entity_type": entity_type,
            "ticker": ticker,
            "company": company,
            "sentiment_score": sent_score,
            "sentiment_label": sent_label,
            "event_type": event_type,
            "event_confidence": event_conf,
            "impact_z": np.nan,
            "impact_score": 5, # neutral masked default
            "has_sentiment_label": True,
            "has_event_label": True,
            "has_impact_label": False
        })
        
    # 2. Ingest Benzinga News (Dated Headlines + Impact)
    bz_path = "data/raw/benzinga/analyst_ratings_processed.csv"
    print("Sampling Benzinga News headlines...")
    df_bz = pd.read_csv(bz_path, nrows=80000).dropna(subset=["title", "date", "stock"])
    
    # Filter to universe tickers for high quality event study
    universe_tickers = set(linker.ticker_to_entity.keys())
    df_bz_universe = df_bz[df_bz["stock"].str.upper().isin(universe_tickers)]
    print(f"Found {len(df_bz_universe)} Benzinga headlines belonging to S&P 100 universe.")
    
    bz_sample = df_bz_universe.sample(n=min(len(df_bz_universe), 4000), random_state=42)
    z_scores_bz = []
    
    for idx, row in bz_sample.iterrows():
        text = str(row["title"]).strip()
        ticker = str(row["stock"]).strip().upper()
        date_str = str(row["date"]).strip()
        company = linker.ticker_to_entity[ticker]["name"]
        
        try:
            pub_date = pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            continue
            
        z = impact_engine.compute_abnormal_move(ticker, pub_date)
        z_scores_bz.append(z)
        
        event_type, event_conf = labeler.label(text)
        
        records.append({
            "doc_id": f"benzinga_{idx}",
            "source": "benzinga_news",
            "source_type": "news",
            "published_at": pub_date.isoformat(),
            "text": text,
            "entity_type": "company",
            "ticker": ticker,
            "company": company,
            "sentiment_score": 0.0, # masked sentiment for raw headlines
            "sentiment_label": "neutral",
            "event_type": event_type,
            "event_confidence": event_conf,
            "impact_z": z,
            "impact_score": 5, # will calibrate
            "has_sentiment_label": False,
            "has_event_label": True,
            "has_impact_label": (z is not None and not np.isnan(z))
        })
        
    # 3. Fit empirical impact decile thresholds on observed z-scores
    valid_z = [r["impact_z"] for r in records if not np.isnan(r["impact_z"])]
    print(f"Observed {len(valid_z)} valid event-study abnormal moves for decile calibration.")
    edges = impact_engine.fit_bins(valid_z)
    
    # Save binning configuration
    os.makedirs("config", exist_ok=True)
    with open("config/impact_bins.yaml", "w") as f:
        yaml.dump({"bin_edges": [float(x) for x in edges]}, f)
    print("Saved calibrated impact bin edges to config/impact_bins.yaml")
    
    # Map impact scores
    for r in records:
        if r["has_impact_label"]:
            r["impact_score"] = impact_engine.map_to_score(r["impact_z"])
            
    df_all = pd.DataFrame(records)
    out_dir = "data/samples"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "unified_risk_dataset.csv")
    df_all.to_csv(out_file, index=False)
    print(f"\nSuccessfully built unified dataset with {len(df_all)} instances at {out_file}")
    print("\nModality Breakdown:")
    print(df_all["source"].value_counts())
    print("\nImpact Score Distribution:")
    print(df_all[df_all["has_impact_label"]]["impact_score"].value_counts().sort_index())

if __name__ == "__main__":
    build_dataset()
