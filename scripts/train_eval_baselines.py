import json
import os

import pandas as pd

from src.eval.metrics import event_metrics, impact_metrics, sentiment_metrics
from src.models.baselines import BaselineModelSuite

DATASET = "data/samples/unified_risk_dataset.csv"


def run_baseline_evaluation(dataset_path: str = DATASET, out_dir: str = "models/baseline"):
    """Trains the TF-IDF baselines on the train split and scores them on the shared test split."""
    df = pd.read_csv(dataset_path)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    suite = BaselineModelSuite()
    results = {}

    s_tr, s_te = train[train["has_sentiment_label"]], test[test["has_sentiment_label"]]
    suite.fit_sentiment(s_tr["text"].tolist(), s_tr["sentiment_score"].tolist())
    results["sentiment"] = {
        "model": "TF-IDF + Ridge",
        **sentiment_metrics(s_te["sentiment_score"].values, suite.predict_sentiment(s_te["text"].tolist())),
    }

    e_tr, e_te = train[train["has_event_label"]], test[test["has_event_label"]]
    suite.fit_event(e_tr["text"].tolist(), e_tr["event_type"].tolist())
    results["event_type"] = {
        "model": "TF-IDF + LogisticRegression",
        **event_metrics(e_te["event_type"].tolist(), list(suite.predict_event(e_te["text"].tolist()))),
    }

    i_tr, i_te = train[train["has_impact_label"]], test[test["has_impact_label"]]
    suite.fit_impact(i_tr["text"].tolist(), i_tr["impact_score"].tolist())
    results["impact_score"] = {
        "model": "TF-IDF + Ridge",
        **impact_metrics(i_te["impact_score"].values, suite.predict_impact(i_te["text"].tolist())),
    }

    suite.save(out_dir)
    os.makedirs("reports", exist_ok=True)
    with open("reports/baselines_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: {m: v for m, v in r.items() if m not in ("confusion", "per_class")}
                      for k, r in results.items()}, indent=2))
    return suite, results


if __name__ == "__main__":
    run_baseline_evaluation()
