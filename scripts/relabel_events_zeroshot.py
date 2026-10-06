"""
Replaces circular keyword supervision with NLI zero-shot event labels.

Writes data/labels/event_labels_zeroshot.csv (one row per text, resumable), then merges the
reconciled label back into data/samples/unified_risk_dataset.csv:

  event_type / event_confidence   reconciled label used for training
  event_type_zs / event_conf_zs   raw zero-shot prediction
  event_label_source              agree | zeroshot | rule | default
  has_event_label                 True only where a zero-shot pass exists

Rows are processed in priority order (hand-label eval set, then train, val, test), so a
partially completed run still yields usable supervision.

Usage: python -m scripts.relabel_events_zeroshot [--merge-only]
"""
import argparse
import json
import os

import pandas as pd

from src.labeling.zeroshot import LABELS, ZeroShotEventClassifier, resolve_event_label

DATASET = "data/samples/unified_risk_dataset.csv"
HAND = "data/labels/hand_labelled_eval.csv"
OUT = "data/labels/event_labels_zeroshot.csv"
CHECKPOINT_EVERY = 400


BUDGET = {"train": 6000, "val": 1500, "test": 2500}


def select_split(df: pd.DataFrame, split: str, budget: int, seen: set, seed: int = 42) -> list:
    """
    Every row where the keyword rule fires is kept (they carry the rare classes); the rest of
    the budget is a seeded random sample, so the labelled subset stays representative.
    Already-labelled texts count towards the budget.
    """
    part = df[df["split"] == split].drop_duplicates("text").sample(frac=1.0, random_state=seed)
    fired = part[part["event_type_rule"] != "Other/None"]["text"].astype(str).tolist()
    rest = part[part["event_type_rule"] == "Other/None"]["text"].astype(str).tolist()
    chosen = [t for t in part["text"].astype(str) if t in seen]
    for t in fired + rest:
        if len(chosen) >= budget:
            break
        if t not in seen and t not in chosen:
            chosen.append(t)
    return chosen


def label_texts():
    df = pd.read_csv(DATASET)
    hand = pd.read_csv(HAND)
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["text"])
    seen = set(done["text"].astype(str))

    queue = [*hand["text"].astype(str).tolist()]
    for split in ("train", "val", "test"):
        queue += select_split(df, split, BUDGET[split], seen)
    queue = list(dict.fromkeys(queue))
    todo = [t for t in queue if t not in seen]
    print(f"{len(seen)} texts already labelled, {len(todo)} remaining (budget {BUDGET})")
    if not todo:
        return

    clf = ZeroShotEventClassifier()
    print(f"Zero-shot model on {clf.device}")
    for start in range(0, len(todo), CHECKPOINT_EVERY):
        batch = todo[start:start + CHECKPOINT_EVERY]
        probs = clf.predict_proba(batch)
        rows = pd.DataFrame(probs, columns=[f"p_{l}" for l in LABELS])
        rows.insert(0, "text", batch)
        rows["event_type_zs"] = [LABELS[i] for i in probs.argmax(axis=1)]
        rows["event_conf_zs"] = probs.max(axis=1)
        rows.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"  labelled {start + len(batch)}/{len(todo)}", flush=True)


def merge():
    df = pd.read_csv(DATASET)
    zs = pd.read_csv(OUT)[["text", "event_type_zs", "event_conf_zs"]].drop_duplicates("text")
    df = df.drop(columns=[c for c in ("event_type_zs", "event_conf_zs", "event_label_source") if c in df])
    df = df.merge(zs, on="text", how="left")

    has_zs = df["event_type_zs"].notna()
    resolved = [
        resolve_event_label(r, rc, z, zc) if ok else (r, rc, "rule_only")
        for r, rc, z, zc, ok in zip(df["event_type_rule"], df["event_confidence_rule"],
                                    df["event_type_zs"], df["event_conf_zs"], has_zs)
    ]
    df["event_type"] = [r[0] for r in resolved]
    df["event_confidence"] = [round(r[1], 4) for r in resolved]
    df["event_label_source"] = [r[2] for r in resolved]
    df["has_event_label"] = has_zs
    df.to_csv(DATASET, index=False)

    lab = df[has_zs]
    agreement = float((lab["event_type_rule"] == lab["event_type_zs"]).mean())
    non_other = lab[lab["event_type_rule"] != "Other/None"]
    stats = {
        "labelled_rows": int(has_zs.sum()),
        "total_rows": int(len(df)),
        "rule_vs_zeroshot_agreement": round(agreement, 4),
        "agreement_when_rule_fires": round(float((non_other["event_type_rule"] == non_other["event_type_zs"]).mean()), 4),
        "label_source_counts": lab["event_label_source"].value_counts().to_dict(),
        "class_support_by_split": {
            s: lab[lab["split"] == s]["event_type"].value_counts().to_dict() for s in ("train", "val", "test")
        },
        "rule_class_distribution": lab["event_type_rule"].value_counts().to_dict(),
        "final_class_distribution": lab["event_type"].value_counts().to_dict(),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/event_labels.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not args.merge_only:
        label_texts()
    merge()
