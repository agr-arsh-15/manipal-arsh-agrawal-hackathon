"""
Module A backtest: sentiment-tilted 20-name index vs. its equal-weight parent.

    python -m scripts.run_module_a

Strategy parameters (tilt strength, sentiment half-life, impact weighting) are chosen by
information ratio on the in-sample window only (before the out-of-sample start); the
out-of-sample window is evaluated once with the chosen configuration.

Writes reports/module_a_backtest.json plus NAV and weight time series under data/outputs/.
"""
import itertools
import json
import os

import pandas as pd
import yaml

from src.engine.store import SignalStore
from src.modules.rebalancer import RebalancerConfig, run_backtest

START, END = "2018-04-02", "2020-06-10"
GRID = {"tilt_lambda": [1.0, 2.0, 3.0, 5.0], "halflife_days": [2.0, 5.0, 10.0], "impact_weighted": [False, True]}


def select_config(signals, in_sample_end: str):
    rows = []
    for lam, hl, iw in itertools.product(*GRID.values()):
        cfg = RebalancerConfig(tilt_lambda=lam, halflife_days=hl, impact_weighted=iw)
        full = run_backtest(signals, START, in_sample_end, oos_start=in_sample_end, config=cfg)["report"]["full_period"]
        rows.append({"tilt_lambda": lam, "halflife_days": hl, "impact_weighted": iw,
                     "in_sample_information_ratio": full["active"]["information_ratio"],
                     "in_sample_excess_return": full["active"]["excess_total_return"],
                     "in_sample_ic": full["sentiment_ic"]["mean"]})
    best = max(rows, key=lambda r: r["in_sample_information_ratio"])
    return RebalancerConfig(tilt_lambda=best["tilt_lambda"], halflife_days=best["halflife_days"],
                            impact_weighted=best["impact_weighted"]), rows


def main():
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    signals = SignalStore(cfg["paths"]["signals_file"]).read()
    oos_start = cfg["splits"]["val_end"]
    in_sample_end = (pd.Timestamp(oos_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    chosen, grid = select_config(signals, in_sample_end)
    out = run_backtest(signals, START, END, oos_start=oos_start, config=chosen)
    report = out["report"]
    report["model_version"] = signals[0].model_version if signals else None
    report["parameter_selection"] = {
        "criterion": "information ratio on the in-sample window",
        "in_sample": [START, in_sample_end],
        "grid": sorted(grid, key=lambda r: -r["in_sample_information_ratio"]),
    }

    os.makedirs("data/outputs", exist_ok=True)
    out["nav"].to_csv("data/outputs/module_a_nav.csv")
    out["weights"].round(5).to_csv("data/outputs/module_a_weights.csv")
    out["sentiment_state"].round(4).to_csv("data/outputs/module_a_sentiment_state.csv")
    with open("reports/module_a_backtest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Chosen (in-sample):", report["config"])
    print(json.dumps({k: report[k] for k in ("full_period", "out_of_sample")}, indent=2))


if __name__ == "__main__":
    main()
