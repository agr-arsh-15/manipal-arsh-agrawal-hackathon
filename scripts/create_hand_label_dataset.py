import pandas as pd
import numpy as np
import os
import yaml
from src.labeling.rules import EventTaxonomyLabeler

def generate_hand_label_template():
    print("Generating stratified hand-label evaluation set candidate (target >= 220)...")
    labeler = EventTaxonomyLabeler("config/taxonomy.yaml")
    
    # 1. Load Benzinga news headlines
    benzinga_path = "data/raw/benzinga/analyst_ratings_processed.csv"
    df_bz = pd.read_csv(benzinga_path, nrows=100000)
    df_bz = df_bz.dropna(subset=["title", "date", "stock"])
    
    # 2. Load Stock Tweets
    tweets_path = "data/raw/stock_tweets/reduced_dataset-release.csv"
    df_tw = pd.read_csv(tweets_path, nrows=60000)
    df_tw = df_tw.dropna(subset=["TWEET", "DATE", "STOCK"])
    
    candidates = []
    
    # Sample from Benzinga
    for idx, row in df_bz.sample(n=min(len(df_bz), 10000), random_state=42).iterrows():
        title = str(row["title"]).strip()
        stock = str(row["stock"]).strip()
        date = str(row["date"]).strip()
        weak_event, conf = labeler.label(title)
        candidates.append({
            "source": "benzinga_news",
            "source_type": "news",
            "date": date,
            "ticker_hint": stock,
            "text": title,
            "rule_suggested_event": weak_event,
            "user_ground_truth_event": "", # BLANK
            "user_verified_ticker": "",    # BLANK
            "notes": ""
        })
        
    # Sample from Tweets
    for idx, row in df_tw.sample(n=min(len(df_tw), 8000), random_state=42).iterrows():
        text = str(row["TWEET"]).strip()
        stock = str(row["STOCK"]).strip()
        date = str(row["DATE"]).strip()
        weak_event, conf = labeler.label(text)
        candidates.append({
            "source": "stock_tweets",
            "source_type": "social",
            "date": date,
            "ticker_hint": stock,
            "text": text,
            "rule_suggested_event": weak_event,
            "user_ground_truth_event": "", # BLANK
            "user_verified_ticker": "",    # BLANK
            "notes": ""
        })

    df_cand = pd.DataFrame(candidates)
    
    sampled_list = []
    for cls in labeler.class_names:
        sub = df_cand[df_cand["rule_suggested_event"] == cls]
        # Target up to 30 items per class to hit 220+ total
        n_take = min(len(sub), 30)
        if n_take > 0:
            sampled_list.append(sub.sample(n=n_take, random_state=42))

    df_eval = pd.concat(sampled_list).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    out_path = "data/labels/hand_labelled_eval.csv"
    os.makedirs("data/labels", exist_ok=True)
    df_eval.to_csv(out_path, index=False)
    print(f"Generated clean hand-labeling template with {len(df_eval)} instances at: {out_path}")
    print("\nClass breakdown in generated template:")
    print(df_eval["rule_suggested_event"].value_counts())

if __name__ == "__main__":
    generate_hand_label_template()
