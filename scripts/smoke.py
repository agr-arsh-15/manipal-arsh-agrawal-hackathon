"""
End-to-end smoke check, run by CI on Windows, macOS and Linux.

    python run.py smoke [--backend auto|baseline|transformer] [--skip-dashboard]

Checks, in order: the engine scores demo headlines into schema-valid signals; a Module B
scenario runs and lowers CET1; a Module A backtest runs on the committed signal stream; the
committed reports and docs load; the dashboard script executes without exceptions and the
Streamlit server answers its health endpoint.
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

SIGNALS = "data/samples/signals.jsonl"
DEMO = [
    "Russia launches invasion as Western allies prepare sweeping sanctions",
    "Federal Reserve signals 75 basis point rate hike to fight surging inflation",
    "Moody's downgrades Boeing to junk as cash burn accelerates",
    "Apple shares hit record after iPhone revenue beats analyst estimates",
    "$TSLA recalls 360,000 vehicles over self-driving software defect",
]
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
    return f"{engine.backend} backend ({engine.model_version}), {len(signals)} signals"


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


def check_dashboard_script() -> str:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.abspath("app/dashboard.py"), default_timeout=300).run()
    if at.exception:
        raise AssertionError("; ".join(str(e.value) for e in at.exception))
    return f"script ran cleanly ({len(at.tabs)} tabs)"


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
        steps += [("dashboard script", check_dashboard_script), ("dashboard server", check_dashboard_server)]

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
