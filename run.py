#!/usr/bin/env python
"""
Cross-platform task runner (Windows, macOS, Linux). Uses only the standard library.

    python run.py <task> [extra args passed through]

Tasks:
    fetch-model   download the fine-tuned transformer from the GitHub Release
    test          pytest suite
    smoke         end-to-end check: engine, Module A, Module B, reports, dashboard
    dashboard     Streamlit UI on http://localhost:8501
    api           FastAPI on http://127.0.0.1:8000 (interactive docs at /docs)
    evaluate      baseline vs transformer report + figures
    signals       re-score the full signal stream
    modules       synthetic book, Module A backtest, Module B scenarios
    deck          slide deck (+ architecture.png when Graphviz is installed)
    train         fine-tune the multi-task transformer
    data          rebuild the labelled dataset from data/raw (needs Kaggle downloads)
    label         zero-shot event labelling with BART-MNLI

Run it with the interpreter of your virtual environment (activate it first).
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(ROOT, "models", "risk_engine", "heads.pt")
REQUIRED_MODULES = ["numpy", "pandas", "yaml", "sklearn", "torch", "transformers",
                    "fastapi", "uvicorn", "streamlit", "plotly", "matplotlib", "reportlab", "pytest"]

# Inference tasks run against local files only; train/label must reach the Hugging Face Hub
# the first time to download the base models.
OFFLINE_TASKS = {"test", "smoke", "dashboard", "api", "evaluate", "signals", "modules", "deck"}
MODEL_TASKS = {"smoke", "dashboard", "api", "evaluate", "signals"}


def task_env(task: str) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "MPLCONFIGDIR": os.path.join(ROOT, ".cache", "matplotlib"),
    })
    if task in OFFLINE_TASKS:
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def preflight(task: str) -> None:
    if sys.version_info[:2] != (3, 11):
        print(f"[warn] Python {sys.version.split()[0]} detected; the project is pinned and tested on 3.11.")
    if sys.prefix == sys.base_prefix and not os.environ.get("CI"):
        print("[warn] Not running inside a virtual environment. See README section 4.")
    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"[error] Missing packages: {', '.join(missing)}\n"
                 f"        Install them with:  {os.path.basename(sys.executable)} -m pip install -r requirements.txt")
    if task in MODEL_TASKS and not os.path.exists(CHECKPOINT):
        print("[info] Fine-tuned transformer not found; the TF-IDF baseline will be used.\n"
              "       Get the transformer with:  python run.py fetch-model")


def py(*args: str) -> list:
    return [sys.executable, *args]


def commands(task: str, extra: list) -> list:
    if task == "fetch-model":
        return [py("-m", "scripts.fetch_model", *extra)]
    if task == "test":
        return [py("-m", "pytest", "tests", "-q", *extra)]
    if task == "smoke":
        return [py("-m", "scripts.smoke", *extra)]
    if task == "dashboard":
        return [py("-m", "streamlit", "run", "app/dashboard.py", "--browser.gatherUsageStats", "false", *extra)]
    if task == "api":
        return [py("-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000", *extra)]
    if task == "evaluate":
        return [py("-m", "scripts.evaluate_engine", *extra)]
    if task == "signals":
        return [py("-m", "scripts.generate_signals", *extra)]
    if task == "modules":
        return [py("-m", "scripts.build_synthetic_portfolio"), py("-m", "scripts.run_module_a"),
                py("-m", "scripts.run_module_b")]
    if task == "deck":
        cmds = [py("-m", "scripts.build_deck", *extra)]
        dot = shutil.which("dot")
        if dot:
            cmds.append([dot, "-Tpng", "docs/architecture.dot", "-o", "docs/architecture.png"])
        else:
            print("[info] Graphviz 'dot' not on PATH; keeping the committed docs/architecture.png.")
        return cmds
    if task == "train":
        return [py("-m", "scripts.train_multitask", *extra)]
    if task == "data":
        return [py("-m", "scripts.build_unified_dataset", *extra)]
    if task == "label":
        return [py("-m", "scripts.relabel_events_zeroshot", *extra)]
    raise ValueError(task)


TASKS = ["fetch-model", "test", "smoke", "dashboard", "api", "evaluate", "signals", "modules",
         "deck", "train", "data", "label"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=TASKS)
    args, extra = parser.parse_known_args()

    preflight(args.task)
    env = task_env(args.task)
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    for cmd in commands(args.task, extra):
        print("$", " ".join("python" if c == sys.executable else c for c in cmd), flush=True)
        try:
            code = subprocess.call(cmd, cwd=ROOT, env=env)
        except KeyboardInterrupt:
            return 130
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
