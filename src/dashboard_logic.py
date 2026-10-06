"""
Presentation logic for app/dashboard.py, kept free of Streamlit so it can be unit-tested:
curated demo inputs, trigger-gate explanations, model-card rows and number formatting.
"""
import math
from typing import Dict, List, Optional, Sequence

import pandas as pd
import yaml

# Each headline was scored with the v1.0.0 transformer before being added; the expected outputs
# are pinned in tests/test_dashboard_logic.py so a retrained model cannot silently break the demo.
DEMO_HEADLINES = [
    "Microsoft beats quarterly earnings estimates and raises full-year guidance",
    "Intel shares plunge after weak guidance and delayed chip roadmap",
    "US unemployment surges as recession fears mount; Fed may hold rates",
    "S&P cuts Ford credit rating to junk on weak cash flow",
    "Factory fire halts production at key semiconductor supplier",
    "Disney agrees to acquire 21st Century Fox assets for $71 billion",
]
# A headline from the historical stream that the live engine scores above every trigger gate.
STRESS_DEFAULT_HEADLINE = "US Federal Government Posts Widest Deficit Since 2012"

STRESS_TRIGGERED = "Triggered by the model"
STRESS_FORCED = "What-if scenario: forced run, the trigger rule was not met"
STRESS_NOT_RUN = "Not triggered"

# (task, metric, label, higher_is_better)
MODEL_CARD_METRICS = [
    ("sentiment", "pearson_r", "Sentiment: correlation with label (Pearson r)", True),
    ("sentiment", "macro_f1", "Sentiment: 3-class macro-F1", True),
    ("sentiment", "mae", "Sentiment: mean absolute error", False),
    ("event_type", "macro_f1", "Event type: macro-F1 (9 classes)", True),
    ("event_type", "accuracy", "Event type: accuracy", True),
    ("impact_score", "spearman_rho", "Impact: rank correlation (Spearman)", True),
    ("impact_score", "high_impact_recall", "Impact: recall of high-impact news", True),
    ("impact_score", "high_impact_precision", "Impact: precision on high-impact news", True),
    ("impact_score", "mae", "Impact: mean absolute error (1-10 scale)", False),
    ("impact_score_company", "spearman_rho", "Impact, company news: Spearman", True),
    ("impact_score_market", "spearman_rho", "Impact, market-wide news: Spearman", True),
]


def parse_headlines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def oos_start(config_path: str = "config/engine.yaml") -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        return str(yaml.safe_load(f)["splits"]["val_end"])


def fmt_money(x: float, signed: bool = False) -> str:
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    a = abs(x)
    for scale, suffix in ((1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if a >= scale:
            return f"{sign}${a / scale:,.1f}{suffix}"
    return f"{sign}${a:,.0f}"


def trigger_gates(signal, trigger: Dict) -> List[Dict]:
    """One row per Module B trigger condition, with the signal's value and whether it passes."""
    max_sent = trigger.get("max_sentiment", 1.0)
    min_conf = trigger.get("min_event_confidence", 0.0)
    return [
        {"gate": "Event type is systemic", "rule": ", ".join(trigger["event_types"]),
         "signal": signal.event_type, "passed": signal.event_type in trigger["event_types"]},
        {"gate": "Impact is high", "rule": f">= {trigger['min_impact']}",
         "signal": str(signal.impact_score), "passed": signal.impact_score >= trigger["min_impact"]},
        {"gate": "Sentiment is adverse", "rule": f"< {max_sent:+.2f}",
         "signal": f"{signal.sentiment_score:+.2f}", "passed": signal.sentiment_score < max_sent},
        {"gate": "Event classification is confident", "rule": f">= {min_conf:.0%}",
         "signal": f"{signal.event_confidence:.0%}", "passed": signal.event_confidence >= min_conf},
    ]


def stress_run_label(triggered: bool, forced: bool) -> str:
    if triggered:
        return STRESS_TRIGGERED
    return STRESS_FORCED if forced else STRESS_NOT_RUN


def model_card_rows(ev: Dict) -> pd.DataFrame:
    test = ev.get("test", {})
    base, tf = test.get("baseline", {}), test.get("transformer", {})
    rows = []
    for task, metric, label, higher in MODEL_CARD_METRICS:
        b = base.get(task, {}).get(metric)
        t = tf.get(task, {}).get(metric)
        if b is None and t is None:
            continue
        n = (tf.get(task) or base.get(task) or {}).get("n")
        verdict = None
        if b is not None and t is not None:
            better = t > b if higher else t < b
            verdict = "better" if better else ("same" if math.isclose(t, b) else "worse")
        rows.append({"metric": label, "better when": "higher" if higher else "lower", "test rows": n,
                     "TF-IDF baseline": b, "transformer": t, "transformer vs baseline": verdict})
    return pd.DataFrame(rows)


def hand_label_summary(ev: Dict) -> Optional[pd.DataFrame]:
    """Human-labelled results as a table, or None while the labelling is not done."""
    hl = ev.get("hand_labelled") or {}
    if hl.get("status") not in ("complete", "partial"):
        return None
    rows = [{"approach": name, **vals} for name, vals in hl.items() if isinstance(vals, dict)]
    return pd.DataFrame(rows) if rows else None


def stream_scope(sig_df: pd.DataFrame) -> List[str]:
    """Human-readable date coverage of each source in the signal stream."""
    if sig_df.empty:
        return []
    out = []
    for source, g in sig_df.groupby("source"):
        lo, hi = g["timestamp"].min(), g["timestamp"].max()
        span = f"{lo:%b %Y}" if (lo.year, lo.month) == (hi.year, hi.month) else f"{lo:%b %Y} to {hi:%b %Y}"
        out.append(f"{source}: {len(g):,} signals, {span}")
    return out


def result_for_download(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if k != "positions"}


def cet1_breach_count(triggers: pd.DataFrame, minimum: float) -> int:
    return int((triggers["cet1_ratio_after"] < minimum).sum()) if "cet1_ratio_after" in triggers else 0


def gate_failures(gates: Sequence[Dict]) -> List[str]:
    return [f"{g['gate']} ({g['signal']}, needs {g['rule']})" for g in gates if not g["passed"]]


def calibration_range(calibration: Sequence[Dict]) -> Optional[tuple]:
    """Realised mean severity in the lowest and highest predicted impact buckets."""
    if not calibration:
        return None
    rows = sorted(calibration, key=lambda r: r["predicted_bucket"])
    return rows[0]["realised_mean"], rows[-1]["realised_mean"]
