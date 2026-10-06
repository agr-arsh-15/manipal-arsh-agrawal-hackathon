"""
Module A backtest: sentiment-tilted 20-name index vs. its equal-weight parent.

    python -m scripts.run_module_a

Writes reports/module_a_backtest.json plus NAV and weight time series under data/outputs/.
"""
import json
import os

import yaml

from src.engine.store import SignalStore
from src.modules.rebalancer import run_backtest

START, END = "2018-04-02", "2020-06-10"


def main():
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    signals = SignalStore(cfg["paths"]["signals_file"]).read()
    out = run_backtest(signals, START, END, oos_start=cfg["splits"]["val_end"])
    report = out["report"]
    report["model_version"] = signals[0].model_version if signals else None

    os.makedirs("data/outputs", exist_ok=True)
    out["nav"].to_csv("data/outputs/module_a_nav.csv")
    out["weights"].round(5).to_csv("data/outputs/module_a_weights.csv")
    out["sentiment_state"].round(4).to_csv("data/outputs/module_a_sentiment_state.csv")
    with open("reports/module_a_backtest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("full_period", "out_of_sample")}, indent=2))


if __name__ == "__main__":
    main()
