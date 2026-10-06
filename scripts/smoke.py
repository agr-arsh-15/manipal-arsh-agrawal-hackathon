"""
End-to-end smoke check, run by CI on Windows, macOS and Linux.

    python run.py smoke [--backend auto|baseline|transformer] [--skip-dashboard]

Checks, in order: the engine scores the dashboard's demo headlines into schema-valid signals
(and, on the transformer, the default stress headline passes every trigger gate); a Module B
scenario runs and lowers CET1; a Module A backtest runs on the committed signal stream; the
committed reports and docs load; every dashboard tab and interaction path executes without
exceptions; and the Streamlit server answers its health endpoint.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback

import requests

from src.dashboard_logic import DEMO_HEADLINES, STRESS_DEFAULT_HEADLINE, STRESS_TRIGGERED

SIGNALS = "data/samples/signals.jsonl"
DEMO = DEMO_HEADLINES
ARTIFACTS = ["reports/engine_eval.json", "reports/module_a_backtest.json", "reports/module_b_stress.json",
             "docs/presentation.pdf", "docs/architecture.png"]


def check_engine(backend: str) -> str:
    from src.engine.pipeline import RiskEngine
    from src.schemas import Signal

    engine = RiskEngine(backend=backend)
    if backend != "auto" and engine.backend != backend:
        raise AssertionError(f"requested {backend}, got {engine.backend}")
    signals = engine.analyze_texts(DEMO)
    assert len(signals) >= len(DEMO), "every headline should yield at least one signal"
    for s in signals:
        Signal.model_validate(s.model_dump())
        assert -1 <= s.sentiment_score <= 1 and 1 <= s.impact_score <= 10
    for s in signals:
        print(f"       {s.sentiment_score:+.2f}  {s.event_type:24s} impact {s.impact_score:2d}  "
              f"{s.ticker or '-':5s} {s.text_excerpt[:60]}")
    detail = f"{engine.backend} backend ({engine.model_version}), {len(signals)} signals"
    if engine.backend == "transformer":
        from src.modules.stress import StressTestEngine

        sig = engine.analyze_texts([STRESS_DEFAULT_HEADLINE])[0]
        assert StressTestEngine().should_trigger(sig), f"default stress headline no longer triggers: {sig}"
        detail += "; default stress headline triggers"
    return detail


def check_module_b() -> str:
    from src.modules.stress import StressTestEngine

    engine = StressTestEngine()
    r = engine.run(engine.shock_for("Macroeconomic", 9))
    assert r["cet1_ratio_after"] < r["cet1_ratio_before"], "a severe macro shock must reduce CET1"
    return f"Macroeconomic @9: P&L {r['total_pnl'] / 1e6:,.0f}m, CET1 {r['cet1_ratio_before']:.2%} -> {r['cet1_ratio_after']:.2%}"


def check_module_a() -> str:
    from src.engine.store import SignalStore
    from src.modules.rebalancer import RebalancerConfig, run_backtest

    signals = SignalStore(SIGNALS).read()
    out = run_backtest(signals, "2019-11-01", "2020-06-10", oos_start="2020-01-02",
                       config=RebalancerConfig(impact_weighted=True))
    w = out["weights"]
    assert ((w.sum(axis=1) - 1).abs() < 1e-6).all(), "weights must sum to 1 every day"
    assert out["nav"].notna().all().all()
    return f"{len(signals):,} signals, {len(w)} sessions, weights sum to 1"


def check_artifacts() -> str:
    for path in ARTIFACTS:
        assert os.path.getsize(path) > 0, f"{path} is empty"
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
    return f"{len(ARTIFACTS)} reports and docs present"


def engine_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    from src.engine.pipeline import TRANSFORMER_DIR
    return "transformer" if os.path.exists(os.path.join(TRANSFORMER_DIR, "heads.pt")) else "baseline"


def _page_text(at) -> str:
    kinds = ("markdown", "caption", "info", "warning", "success", "json")
    return "\n".join(str(e.value) for kind in kinds for e in getattr(at, kind))


def _ok(at, step: str):
    if at.exception:
        raise AssertionError(f"{step}: " + "; ".join(str(e.value) for e in at.exception))
    return at


def _button(at, label: str):
    return next(b for b in at.button if b.label == label)


def check_dashboard_script(backend: str) -> str:
    """Walks every tab and the main interaction paths a judge will click through."""
    from streamlit.testing.v1 import AppTest

    at = _ok(AppTest.from_file(os.path.abspath("app/dashboard.py"), default_timeout=300).run(), "initial load")
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"
    text = _page_text(at)
    assert "label_helper" not in text and '"pending"' not in text, "pending human-label status leaked into the UI"
    if backend == "transformer":
        assert any(STRESS_TRIGGERED in s.value for s in at.success), "default stress run should be model-triggered"

    _button(at, "Score headlines").click()
    _ok(at.run(), "score demo headlines")
    assert any("signal(s)" in c.value for c in at.caption), "scoring the demo should show results"

    at.text_area[0].set_value("   ")
    _button(at, "Score headlines").click()
    _ok(at.run(), "score empty input")
    assert any("Enter at least one headline" in w.value for w in at.warning)

    at.slider(key="p_lambda").set_value(8.0)
    _ok(at.run(), "change Module A parameter")
    assert any("differ from the in-sample selection" in w.value for w in at.warning)
    _button(at, "Reset to selected parameters").click()
    _ok(at.run(), "reset Module A parameters")

    next(t for t in at.text_input if t.label == "Headline").set_value(
        "Apple shares hit record after iPhone revenue beats analyst estimates")
    _button(at, "Run stress test").click()
    _ok(at.run(), "non-triggering stress headline")
    assert any("Not triggered" in i.value for i in at.info)
    next(c for c in at.checkbox if c.label.startswith("Run the scenario even")).check()
    _button(at, "Run stress test").click()
    _ok(at.run(), "forced what-if stress run")
    assert any("What-if" in w.value for w in at.warning)

    trigger_source = next(r for r in at.radio if r.label == "Trigger source")
    trigger_source.set_value("Historical triggered signal")
    _ok(at.run(), "historical stress replay")
    trigger_source = next(r for r in at.radio if r.label == "Trigger source")
    trigger_source.set_value("Manual scenario")
    _ok(at.run(), "manual stress scenario")
    return f"{len(at.tabs)} tabs and 9 interaction paths ran cleanly"


def check_dashboard_server(timeout_s: int = 120) -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/dashboard.py", "--server.headless", "true",
         "--server.port", str(port), "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"streamlit exited with code {proc.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).status_code == 200:
                    return f"server healthy on port {port}"
            except requests.RequestException:
                pass
            time.sleep(1)
        raise AssertionError(f"no health response within {timeout_s}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end smoke check.")
    parser.add_argument("--backend", default="auto", choices=["auto", "baseline", "transformer"])
    parser.add_argument("--skip-dashboard", action="store_true")
    args = parser.parse_args()
    if args.backend != "auto":
        os.environ["RISK_ENGINE_BACKEND"] = args.backend

    steps = [("engine", lambda: check_engine(args.backend)), ("module B", check_module_b),
             ("module A", check_module_a), ("artifacts", check_artifacts)]
    if not args.skip_dashboard:
        steps += [("dashboard script", lambda: check_dashboard_script(engine_backend(args.backend))),
                  ("dashboard server", check_dashboard_server)]

    failed = 0
    for name, fn in steps:
        t0 = time.time()
        try:
            detail = fn()
            print(f"[ ok ] {name}: {detail} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            failed += 1
            print(f"[FAIL] {name}", flush=True)
            traceback.print_exc()
    print(f"\n{len(steps) - failed}/{len(steps)} smoke checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
