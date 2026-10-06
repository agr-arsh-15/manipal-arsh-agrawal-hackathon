"""
Module B: reference stress scenarios plus a replay of the historical signal stream.

    python -m scripts.run_module_b

Writes reports/module_b_stress.json and data/outputs/module_b_triggers.csv.
"""
import json
import os

import yaml

from src.engine.store import SignalStore
from src.modules.stress import StressTestEngine

REFERENCE_IMPACT = 9


def strip(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "positions"}


def main():
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    engine = StressTestEngine()
    p = engine.portfolio

    reference = {
        et: strip(engine.run(engine.shock_for(et, REFERENCE_IMPACT)))
        for et in engine.trigger["event_types"]
    }

    signals = SignalStore(cfg["paths"]["signals_file"]).read()
    triggers = engine.scan(signals)
    os.makedirs("data/outputs", exist_ok=True)
    if not triggers.empty:
        triggers.to_csv("data/outputs/module_b_triggers.csv", index=False)

    worst = None
    if not triggers.empty:
        row = triggers.nsmallest(1, "total_pnl").iloc[0]
        sig = next(s for s in signals if s.signal_id == row["signal_id"])
        worst = strip(engine.run_for_signal(sig))

    report = {
        "portfolio": {
            "positions": int(len(p)),
            "by_asset_class": p.groupby("asset_class")["notional"].agg(["count", "sum"]).astype(float).to_dict(orient="index"),
        },
        "trigger_rule": engine.trigger,
        "reference_scenarios_at_impact": REFERENCE_IMPACT,
        "reference_scenarios": reference,
        "historical_replay": {
            "signals_scanned": len(signals),
            "stress_tests_triggered": int(len(triggers)),
            "by_event_type": triggers["event_type"].value_counts().to_dict() if not triggers.empty else {},
            "worst_event": worst,
        },
    }
    with open("reports/module_b_stress.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    for et, r in reference.items():
        print(f"{et:26s} P&L {r['total_pnl'] / 1e6:9.1f}m ({r['pnl_pct_of_value']:.2%})  "
              f"ECL {r['ecl_before'] / 1e6:6.1f}m -> {r['ecl_after'] / 1e6:6.1f}m  "
              f"CET1 {r['cet1_ratio_before']:.2%} -> {r['cet1_ratio_after']:.2%}  stage2={r['loans_moved_to_stage2']}")
    print(f"Historical replay: {len(triggers)} stress tests triggered from {len(signals)} signals")


if __name__ == "__main__":
    main()
