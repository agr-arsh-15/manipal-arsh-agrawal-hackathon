import os
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import event_metrics, impact_metrics, sentiment_metrics

HAND = "data/labels/hand_labelled_eval.csv"
ZS = "data/labels/event_labels_zeroshot.csv"


def taxonomy_labels() -> List[str]:
    with open("config/taxonomy.yaml", "r", encoding="utf-8") as f:
        return [c["name"] for c in yaml.safe_load(f)["classes"]]


def evaluate_predictor(predict: Callable[[List[str]], Dict[str, np.ndarray]], test: pd.DataFrame) -> Dict:
    """Scores a predictor on each task using only rows that carry that task's label."""
    labels = taxonomy_labels()
    out = {}
    s = test[test["has_sentiment_label"]]
    ps = predict(s["text"].tolist())
    out["sentiment"] = sentiment_metrics(s["sentiment_score"].values, ps["sentiment"])

    e = test[test["has_event_label"]]
    pe = predict(e["text"].tolist())
    out["event_type"] = event_metrics(e["event_type"].tolist(), list(pe["event_type"]), labels)

    i = test[test["has_impact_label"]]
    pi = predict(i["text"].tolist())
    out["impact_score"] = impact_metrics(i["impact_score"].values, np.clip(np.round(pi["impact_raw"]), 1, 10))
    out["_impact_predictions"] = pd.DataFrame({"pred": pi["impact_raw"], "true": i["impact_score"].values,
                                               "conf": pi.get("impact_confidence", np.full(len(i), np.nan))})
    return out


def impact_calibration(pred_df: pd.DataFrame) -> List[Dict]:
    df = pred_df.assign(predicted_bucket=np.clip(np.round(pred_df["pred"]), 1, 10).astype(int))
    agg = df.groupby("predicted_bucket")["true"].agg(["mean", "count"]).reset_index()
    return [{"predicted_bucket": int(r.predicted_bucket), "realised_mean": round(float(r["mean"]), 3), "n": int(r["count"])}
            for _, r in agg.iterrows()]


def hand_labelled_eval(predictors: Dict[str, Callable]) -> Dict:
    """Event accuracy against the human ground truth, for every labelling approach."""
    if not os.path.exists(HAND):
        return {"status": "missing"}
    hand = pd.read_csv(HAND, dtype=str).fillna("")
    hand = hand[hand["user_ground_truth_event"] != ""]
    if hand.empty:
        return {"status": "pending", "labelled_rows": 0,
                "note": "Run `streamlit run scripts/label_helper.py` to label the 268-row set."}
    labels = taxonomy_labels()
    truth = hand["user_ground_truth_event"].tolist()
    texts = hand["text"].tolist()
    res = {"status": "complete" if len(hand) >= 260 else "partial", "labelled_rows": int(len(hand))}
    res["keyword_rules"] = _summary(event_metrics(truth, hand["rule_suggested_event"].tolist(), labels))
    if os.path.exists(ZS):
        zs = pd.read_csv(ZS).drop_duplicates("text").set_index("text")["event_type_zs"]
        res["zero_shot_bart_mnli"] = _summary(event_metrics(truth, [zs.get(t, "Other/None") for t in texts], labels))
    for name, predict in predictors.items():
        res[name] = _summary(event_metrics(truth, list(predict(texts)["event_type"]), labels))
    return res


def _summary(m: Dict) -> Dict:
    return {k: m[k] for k in ("n", "macro_f1", "weighted_f1", "accuracy")}


def latency(predict: Callable, text: str, runs: int = 30, batch: int = 64) -> Dict:
    predict([text])
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict([text])
        times.append((time.perf_counter() - t0) * 1000)
    texts = [text] * batch
    t0 = time.perf_counter()
    for _ in range(3):
        predict(texts)
    throughput = 3 * batch / (time.perf_counter() - t0)
    return {"single_p50_ms": round(float(np.percentile(times, 50)), 2),
            "single_p95_ms": round(float(np.percentile(times, 95)), 2),
            "batch_throughput_per_s": round(float(throughput), 1)}
