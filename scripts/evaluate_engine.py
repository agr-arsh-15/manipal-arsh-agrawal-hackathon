"""
Gate 4 evaluation: baseline vs. multi-task transformer on the shared chronological test split,
plus the human-labelled event set, impact calibration and inference latency.

    python -m scripts.evaluate_engine

Writes reports/engine_eval.json, reports/engine_eval.md and docs/figures/*.png.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.train_eval_baselines import run_baseline_evaluation  # noqa: E402
from src.engine.pipeline import _BaselineBackend  # noqa: E402
from src.eval.evaluate import evaluate_predictor, hand_labelled_eval, impact_calibration, latency  # noqa: E402

DATASET = "data/samples/unified_risk_dataset.csv"
FIG_DIR = "docs/figures"
SAMPLE_TEXT = "Federal Reserve signals further rate hikes as inflation stays elevated"


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def plot_comparison(results: dict, path: str):
    tasks = [("sentiment", "pearson_r", "Sentiment\nPearson r"),
             ("event_type", "macro_f1", "Event type\nmacro-F1"),
             ("impact_score", "spearman_rho", "Impact\nSpearman ρ"),
             ("impact_score", "high_impact_precision", "Impact > 7\nprecision")]
    models = [m for m in ("baseline", "transformer") if m in results]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, m in enumerate(models):
        vals = [results[m][t][k] for t, k, _ in tasks]
        bars = ax.bar(x + (j - 0.5 * (len(models) - 1)) * 0.38, vals, 0.36,
                      label={"baseline": "TF-IDF + linear", "transformer": "Multi-task DistilRoBERTa"}[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [t[2] for t in tasks])
    ax.set_ylim(0, 1)
    ax.set_title("Held-out chronological test split")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(m: dict, path: str):
    cm = np.array(m["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = [l.replace("/", "/\n") for l in m["labels"]]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("label")
    ax.set_title("Event classification (test split, row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(cal: list, path: str):
    df = pd.DataFrame(cal)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([1, 10], [1, 10], "--", color="gray", label="ideal")
    ax.plot(df["predicted_bucket"], df["realised_mean"], "o-", label="transformer")
    for _, r in df.iterrows():
        ax.annotate(f"n={r['n']}", (r["predicted_bucket"], r["realised_mean"]), fontsize=7,
                    textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("predicted impact score")
    ax.set_ylabel("mean realised impact score")
    ax.set_title("Impact calibration (test split)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_markdown(report: dict, path: str):
    t = report["test"]
    models = [m for m in ("baseline", "transformer") if m in t]
    lines = ["# Engine evaluation (Gate 4)", "",
             f"Test split: chronological (Benzinga published on or after {report['splits']['val_end']}) plus a "
             "hash-held-out 15% of FinancialPhraseBank. Event labels on the test split are BART-MNLI zero-shot "
             "labels reconciled with keyword rules; the human-labelled set below is the independent check.", "",
             "| Task | Metric | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for task, metrics in [("sentiment", ["n", "pearson_r", "mae", "macro_f1"]),
                          ("event_type", ["n", "macro_f1", "weighted_f1", "accuracy"]),
                          ("impact_score", ["n", "spearman_rho", "mae", "high_impact_precision",
                                            "high_impact_recall", "high_impact_base_rate"])]:
        for m in metrics:
            lines.append(f"| {task} | {m} | " + " | ".join(str(t[k][task].get(m)) for k in models) + " |")
    lines += ["", "## Human-labelled event set", "", "```json", json.dumps(report["hand_labelled"], indent=2), "```",
              "", "## Latency (single headline)", "", "```json", json.dumps(report["latency"], indent=2), "```",
              "", "## Impact calibration", "", "| predicted | realised mean | n |", "|---|---|---|"]
    lines += [f"| {c['predicted_bucket']} | {c['realised_mean']} | {c['n']} |" for c in report.get("impact_calibration", [])]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(DATASET)
    test = df[df["split"] == "test"]

    run_baseline_evaluation(DATASET)
    baseline = _BaselineBackend("models/baseline")
    predictors = {"baseline": baseline.predict}

    has_transformer = os.path.exists("models/risk_engine/heads.pt")
    if has_transformer:
        from src.models.multitask import RiskModelPredictor
        transformer = RiskModelPredictor("models/risk_engine")
        predictors["transformer"] = transformer.predict

    results = {name: evaluate_predictor(p, test) for name, p in predictors.items()}
    import yaml
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)["splits"]

    report = {
        "splits": splits,
        "test": {name: {task: strip(m) for task, m in r.items() if not task.startswith("_")} for name, r in results.items()},
        "hand_labelled": hand_labelled_eval(predictors),
        "latency": {"baseline_cpu": latency(baseline.predict, SAMPLE_TEXT)},
    }
    if has_transformer:
        report["latency"][f"transformer_{transformer.device}"] = latency(transformer.predict, SAMPLE_TEXT)
        cpu = RiskModelPredictor("models/risk_engine", device="cpu")
        report["latency"]["transformer_cpu"] = latency(cpu.predict, SAMPLE_TEXT)
        report["impact_calibration"] = impact_calibration(results["transformer"]["_impact_predictions"])
        with open("models/risk_engine/config.json", "r", encoding="utf-8") as f:
            report["transformer_checkpoint"] = json.load(f)
        plot_confusion(report["test"]["transformer"]["event_type"], f"{FIG_DIR}/event_confusion.png")
        plot_calibration(report["impact_calibration"], f"{FIG_DIR}/impact_calibration.png")
    plot_comparison(report["test"], f"{FIG_DIR}/model_comparison.png")

    with open("reports/engine_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, "reports/engine_eval.md")
    print(open("reports/engine_eval.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
