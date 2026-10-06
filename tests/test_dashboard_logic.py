import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.dashboard_logic import (
    DEMO_HEADLINES, STRESS_DEFAULT_HEADLINE, STRESS_FORCED, STRESS_NOT_RUN, STRESS_TRIGGERED,
    calibration_range, cet1_breach_count, fmt_money, gate_failures, hand_label_summary, model_card_rows,
    oos_start, parse_headlines, result_for_download, stream_scope, stress_run_label, trigger_gates,
)
from src.engine.pipeline import TRANSFORMER_DIR, RiskEngine
from src.modules.stress import StressTestEngine

HAS_TRANSFORMER = os.path.exists(os.path.join(TRANSFORMER_DIR, "heads.pt"))
needs_transformer = pytest.mark.skipif(not HAS_TRANSFORMER, reason="fine-tuned checkpoint not downloaded")

# Expected transformer (v1.0.0) outputs for the curated demo: (event_type, sentiment sign, ticker).
PINNED = {
    "Microsoft beats quarterly earnings estimates and raises full-year guidance": ("Earnings/Guidance", 1, "MSFT"),
    "Intel shares plunge after weak guidance and delayed chip roadmap": ("Earnings/Guidance", -1, "INTC"),
    "US unemployment surges as recession fears mount; Fed may hold rates": ("Macroeconomic", -1, None),
    "S&P cuts Ford credit rating to junk on weak cash flow": ("Credit Event", -1, None),
    "Factory fire halts production at key semiconductor supplier": ("Operational/Supply-chain", -1, None),
    "Disney agrees to acquire 21st Century Fox assets for $71 billion": ("Merger/Acquisition", 1, "DIS"),
}

TRIGGER = {"min_impact": 8, "event_types": ["Macroeconomic", "Geopolitical"], "max_sentiment": 0.0,
           "min_event_confidence": 0.6}


def _sig(event_type="Macroeconomic", impact=9, sentiment=-0.5, confidence=0.9):
    return SimpleNamespace(event_type=event_type, impact_score=impact, sentiment_score=sentiment,
                           event_confidence=confidence)


@pytest.fixture(scope="module")
def transformer():
    return RiskEngine(backend="transformer")


def test_demo_defaults_are_curated():
    assert DEMO_HEADLINES == list(PINNED)
    assert len(set(e for e, _, _ in PINNED.values())) >= 5, "demo should cover several event types"
    assert STRESS_DEFAULT_HEADLINE.strip()


@needs_transformer
def test_demo_headlines_score_as_pinned(transformer):
    signals = transformer.analyze_texts(DEMO_HEADLINES)
    assert len(signals) == len(DEMO_HEADLINES)
    for s, text in zip(signals, DEMO_HEADLINES):
        event, sign, ticker = PINNED[text]
        assert s.event_type == event, text
        assert s.sentiment_score * sign > 0, text
        assert s.ticker == ticker, text


@needs_transformer
def test_stress_default_headline_triggers(transformer):
    sig = transformer.analyze_texts([STRESS_DEFAULT_HEADLINE])[0]
    stress = StressTestEngine()
    assert stress.should_trigger(sig)
    assert all(g["passed"] for g in trigger_gates(sig, stress.trigger))


def test_trigger_gates_explain_each_condition():
    gates = trigger_gates(_sig(), TRIGGER)
    assert [g["gate"] for g in gates] == ["Event type is systemic", "Impact is high", "Sentiment is adverse",
                                          "Event classification is confident"]
    assert all(g["passed"] for g in gates)
    assert gate_failures(gates) == []


def test_trigger_gates_report_failures():
    gates = trigger_gates(_sig(event_type="Earnings/Guidance", impact=5, sentiment=0.4, confidence=0.3), TRIGGER)
    assert not any(g["passed"] for g in gates)
    failures = gate_failures(gates)
    assert failures[0] == "Event type is systemic (Earnings/Guidance, needs Macroeconomic, Geopolitical)"
    assert failures[1] == "Impact is high (5, needs >= 8)"
    assert len(failures) == 4


def test_trigger_gates_match_stress_engine_rule():
    stress = StressTestEngine()
    for sig in [_sig(), _sig(impact=7), _sig(sentiment=0.1), _sig(confidence=0.5), _sig(event_type="Earnings/Guidance")]:
        sig.ticker = None
        assert all(g["passed"] for g in trigger_gates(sig, stress.trigger)) == stress.should_trigger(sig)


def test_parse_headlines_drops_blank_lines():
    assert parse_headlines("") == []
    assert parse_headlines(None) == []
    assert parse_headlines("  \n\n  ") == []
    assert parse_headlines(" a \n\nb\n") == ["a", "b"]


def test_stress_run_label_distinguishes_forced_runs():
    assert stress_run_label(True, False) == STRESS_TRIGGERED
    assert stress_run_label(True, True) == STRESS_TRIGGERED
    assert stress_run_label(False, True) == STRESS_FORCED
    assert stress_run_label(False, False) == STRESS_NOT_RUN


def test_fmt_money():
    assert fmt_money(-1_234_567_890) == "-$1.2bn"
    assert fmt_money(45_600_000) == "$45.6m"
    assert fmt_money(45_600_000, signed=True) == "+$45.6m"
    assert fmt_money(-7_500) == "-$7.5k"
    assert fmt_money(12) == "$12"
    assert fmt_money(0, signed=True) == "$0"


def test_model_card_rows_compare_against_baseline():
    ev = {"test": {
        "baseline": {"event_type": {"macro_f1": 0.40, "n": 500}, "impact_score": {"mae": 1.5, "n": 500}},
        "transformer": {"event_type": {"macro_f1": 0.78, "n": 500}, "impact_score": {"mae": 2.0, "n": 500}},
    }}
    rows = model_card_rows(ev).set_index("metric")
    assert rows.loc["Event type: macro-F1 (9 classes)", "transformer vs baseline"] == "better"
    assert rows.loc["Impact: mean absolute error (1-10 scale)", "transformer vs baseline"] == "worse"
    assert rows.loc["Impact: mean absolute error (1-10 scale)", "better when"] == "lower"
    assert set(rows["test rows"]) == {500}
    assert model_card_rows({}).empty


def test_model_card_rows_on_committed_report():
    import json
    with open("reports/engine_eval.json", "r", encoding="utf-8") as f:
        rows = model_card_rows(json.load(f))
    assert not rows.empty
    assert set(rows["transformer vs baseline"].dropna()) <= {"better", "same", "worse"}


@pytest.mark.parametrize("status", ["missing", "pending"])
def test_hand_label_summary_hidden_until_labelled(status):
    assert hand_label_summary({"hand_labelled": {"status": status, "note": "todo"}}) is None
    assert hand_label_summary({}) is None


def test_hand_label_summary_when_complete():
    ev = {"hand_labelled": {"status": "complete", "transformer": {"macro_f1": 0.7}, "baseline": {"macro_f1": 0.5}}}
    df = hand_label_summary(ev)
    assert list(df["approach"]) == ["transformer", "baseline"]


def test_stream_scope():
    ts = lambda y, m: datetime(y, m, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({"source": ["news", "news", "gdelt"], "timestamp": [ts(2018, 4), ts(2020, 6), ts(2026, 10)]})
    assert stream_scope(df) == ["gdelt: 1 signals, Oct 2026", "news: 2 signals, Apr 2018 to Jun 2020"]
    assert stream_scope(df.iloc[0:0]) == []


def test_calibration_range_orders_by_bucket():
    cal = [{"predicted_bucket": 10, "realised_mean": 6.5}, {"predicted_bucket": 2, "realised_mean": 5.4},
           {"predicted_bucket": 1, "realised_mean": 5.3}]
    assert calibration_range(cal) == (5.3, 6.5)
    assert calibration_range([]) is None


def test_result_for_download_drops_positions_and_is_json_safe():
    import json
    result = StressTestEngine().run(StressTestEngine().shock_for("Macroeconomic", 9))
    payload = result_for_download(result)
    assert "positions" not in payload and "total_pnl" in payload
    json.dumps(payload, default=str)


def test_cet1_breach_count():
    df = pd.DataFrame({"cet1_ratio_after": [0.05, 0.08, 0.06]})
    assert cet1_breach_count(df, 0.07) == 2
    assert cet1_breach_count(pd.DataFrame(), 0.07) == 0


def test_oos_start_comes_from_config():
    assert oos_start() == "2019-11-01"
