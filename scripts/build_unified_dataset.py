"""
Builds data/samples/unified_risk_dataset.csv across three modalities:

  - FinancialPhraseBank  (news sentences, human sentiment labels, undated)
  - Benzinga headlines    (news, timestamped, impact labels via CAR event study)
  - Stock Tweets          (social, dated, impact labels via CAR event study)
  - Benzinga market-wide headlines (tagged to index / credit ETFs; impact = standardised SPY move)

Each row carries task masks (has_sentiment_label / has_event_label / has_impact_label) so the
multi-task model only learns from supervision that genuinely exists for that row, plus a
`split` column shared by every downstream training and evaluation script.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import yaml

from src.ingestion.adapters import BenzingaAdapter, PhraseBankAdapter, StockTweetsAdapter
from src.labeling.impact import MarketImpactEngine
from src.labeling.rules import EventTaxonomyLabeler
from src.linking.matcher import EntityLinker

OUT_FILE = "data/samples/unified_risk_dataset.csv"
SENTIMENT_LABELS = {1.0: "positive", 0.0: "neutral", -1.0: "negative"}
# 13F-style fund holdings lists ("X Buys A, B, C, Sells D, ...") carry no market information.
HOLDINGS_LIST = r"\b(?:Buys|Sells)\b.*,"


def hash_split(doc_id: str) -> str:
    bucket = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def time_split(ts: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp) -> str:
    if ts < train_end:
        return "train"
    return "val" if ts < val_end else "test"


def build_dataset():
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["engine"]["random_seed"]
    ds_cfg = cfg["dataset"]
    train_end = pd.Timestamp(cfg["splits"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["splits"]["val_end"], tz="UTC")

    linker = EntityLinker("config/universe.yaml")
    labeler = EventTaxonomyLabeler("config/taxonomy.yaml")
    impact_engine = MarketImpactEngine("data/prices", "config/engine.yaml")
    records = []

    # 1. FinancialPhraseBank: gold sentiment, no timestamp, so no impact label.
    pb = PhraseBankAdapter().load_frame()
    print(f"FinancialPhraseBank: {len(pb)} sentences")
    for idx, row in pb.iterrows():
        text = str(row["text"]).strip()
        label = str(row["sentiment"]).strip().lower()
        links = linker.link(text)
        doc_id = f"phrasebank_{idx}"
        records.append({
            "doc_id": doc_id, "source": "financial_phrasebank", "source_type": "news",
            "published_at": "2020-01-01T00:00:00+00:00", "text": text,
            "ticker": links[0][0] if links else None,
            "sentiment_score": PhraseBankAdapter.SENTIMENT_MAPPING.get(label, 0.0),
            "sentiment_label": label, "has_sentiment_label": True,
            "impact_z": np.nan, "split": hash_split(doc_id),
        })

    # 2. Benzinga headlines (two feeds) restricted to the 20-name universe.
    for feed in ds_cfg["benzinga_feeds"]:
        bz = BenzingaAdapter(feed["path"], start_date=ds_cfg["start_date"]).load_frame()
        bz = bz.sample(n=min(len(bz), feed["max_rows"]), random_state=seed)
        print(f"Benzinga {os.path.basename(feed['path'])}: {len(bz)} headlines")
        for _, row in bz.iterrows():
            ts = row["published_at"]
            records.append({
                "doc_id": f"benzinga_{row['feed']}_{row['native_idx']}", "source": "benzinga_news",
                "source_type": "news", "published_at": ts.isoformat(), "text": row["text"],
                "ticker": row["ticker"], "sentiment_score": np.nan, "sentiment_label": None,
                "has_sentiment_label": False,
                "impact_z": impact_engine.compute_abnormal_move(row["ticker"], ts),
                "split": time_split(ts, train_end, val_end),
            })

    # 3. Stock Tweets: social modality. Retweet duplicates are collapsed first.
    tw = StockTweetsAdapter(ds_cfg["tweets_path"]).load_frame()
    tw = tw[tw["published_at"] >= pd.Timestamp(ds_cfg["start_date"], tz="UTC")]
    tw = tw.drop_duplicates(subset=["TWEET"])
    tw = tw.sample(n=min(len(tw), ds_cfg["tweets_max_rows"]), random_state=seed)
    print(f"Stock Tweets: {len(tw)} tweets")
    for _, row in tw.iterrows():
        ts = row["published_at"]
        records.append({
            "doc_id": f"tweet_{row['native_idx']}", "source": "stock_tweets", "source_type": "social",
            "published_at": ts.isoformat(), "text": row["TWEET"], "ticker": row["ticker"],
            "sentiment_score": np.nan, "sentiment_label": None, "has_sentiment_label": False,
            "impact_z": impact_engine.compute_abnormal_move(row["ticker"], ts),
            "split": time_split(ts, train_end, val_end),
        })

    # 4. Market-wide headlines (tagged to index / credit ETFs): impact = standardised SPY move.
    market = pd.concat([
        BenzingaAdapter(feed["path"], start_date=ds_cfg["start_date"], tickers=ds_cfg["market_proxies"]).load_frame()
        for feed in ds_cfg["benzinga_feeds"]
    ])
    market = market[~market["text"].str.contains(HOLDINGS_LIST, regex=True)].drop_duplicates(subset=["text"])
    print(f"Benzinga market-wide headlines: {len(market)}")
    for _, row in market.iterrows():
        ts = row["published_at"]
        records.append({
            "doc_id": f"benzinga_{row['feed']}_{row['native_idx']}", "source": "benzinga_news",
            "source_type": "news", "published_at": ts.isoformat(), "text": row["text"], "ticker": None,
            "sentiment_score": np.nan, "sentiment_label": None, "has_sentiment_label": False,
            "impact_z": impact_engine.compute_market_move(ts), "impact_scope": "market",
            "split": time_split(ts, train_end, val_end),
        })

    df = pd.DataFrame(records)
    df["impact_z"] = pd.to_numeric(df["impact_z"], errors="coerce")
    df["has_impact_label"] = df["impact_z"].notna()
    df["impact_scope"] = df.get("impact_scope", pd.Series(index=df.index, dtype=object)).fillna("company")

    company = {t: e["name"] for t, e in linker.ticker_to_entity.items()}
    df["company"] = df["ticker"].map(company)
    df["entity_type"] = np.where(df["ticker"].notna(), "company", "event")

    # Weak rule label is kept as its own column; relabel_events_zeroshot.py resolves the
    # final event_type used for training.
    rule = df["text"].apply(labeler.label)
    df["event_type_rule"] = rule.str[0]
    df["event_confidence_rule"] = rule.str[1]
    df["event_type"] = df["event_type_rule"]
    df["event_confidence"] = df["event_confidence_rule"]
    df["has_event_label"] = True

    # Impact deciles are fitted per scope on the training period only, so test labels never
    # inform the bins and a 9 means "top-decile move" for both company and market news.
    bins_out = {"fitted_on": f"train split (published_at < {cfg['splits']['train_end']})"}
    df["impact_score"] = np.nan
    for scope in ("company", "market"):
        in_scope = df["has_impact_label"] & (df["impact_scope"] == scope)
        train_z = df.loc[in_scope & (df["split"] == "train"), "impact_z"].tolist()
        edges = impact_engine.fit_bins(train_z)
        bins_out[scope] = {"bin_edges": [float(x) for x in edges], "n_observations": len(train_z)}
        df.loc[in_scope, "impact_score"] = df.loc[in_scope, "impact_z"].map(impact_engine.map_to_score)
    bins_out["bin_edges"] = bins_out["company"]["bin_edges"]
    with open("config/impact_bins.yaml", "w", encoding="utf-8") as f:
        yaml.dump(bins_out, f, sort_keys=False)

    columns = [
        "doc_id", "source", "source_type", "published_at", "split", "text", "entity_type",
        "ticker", "company", "sentiment_score", "sentiment_label", "event_type",
        "event_confidence", "event_type_rule", "event_confidence_rule", "impact_scope", "impact_z",
        "impact_score", "has_sentiment_label", "has_event_label", "has_impact_label",
    ]
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df[columns].to_csv(OUT_FILE, index=False)

    print(f"\nWrote {len(df)} rows to {OUT_FILE}")
    print(pd.crosstab(df["source"], df["split"]))
    print("\nImpact labels by source:", df.groupby("source")["has_impact_label"].sum().to_dict())
    print("Impact score distribution:", df["impact_score"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    build_dataset()
