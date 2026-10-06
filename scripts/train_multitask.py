"""
Gate 4: fine-tunes the multi-task DistilRoBERTa risk model on the unified dataset.

    python -m scripts.train_multitask [--epochs N] [--limit N]

Uses the `split` column from build_unified_dataset.py (chronological for dated sources), so
the test period is never seen during training or model selection. The best epoch by a
validation composite score is saved to models/risk_engine/.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models.multitask import MultiTaskRiskModel, masked_multitask_loss
from src.utils import resolve_device

DATASET = "data/samples/unified_risk_dataset.csv"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RiskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict):
        self.texts = df["text"].astype(str).tolist()
        self.sentiment = df["sentiment_score"].fillna(0.0).astype(np.float32).values
        self.sentiment_mask = df["has_sentiment_label"].astype(bool).values
        self.event = df["event_type"].map(label_to_idx).fillna(0).astype(np.int64).values
        self.event_mask = df["event_mask"].astype(bool).values
        self.impact = df["impact_score"].fillna(5.0).astype(np.float32).values
        self.impact_mask = df["has_impact_label"].astype(bool).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i


def make_collate(ds: RiskDataset, tokenizer, max_length: int):
    def collate(indices):
        enc = tokenizer([ds.texts[i] for i in indices], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        idx = np.array(indices)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "sentiment": torch.from_numpy(ds.sentiment[idx]),
            "sentiment_mask": torch.from_numpy(ds.sentiment_mask[idx]),
            "event": torch.from_numpy(ds.event[idx]),
            "event_mask": torch.from_numpy(ds.event_mask[idx]),
            "impact": torch.from_numpy(ds.impact[idx]),
            "impact_mask": torch.from_numpy(ds.impact_mask[idx]),
        }
    return collate


def cap_majority_class(df: pd.DataFrame, label: str, max_share: float, seed: int) -> pd.Series:
    """
    Keeps every event label except a random subset of the majority class, whose event loss
    is masked so it makes up at most `max_share` of event supervision. Rows stay in the
    batch for their other tasks.
    """
    mask = df["has_event_label"].astype(bool).copy()
    is_major = mask & (df["event_type"] == label)
    n_other = int((mask & ~is_major).sum())
    keep = int(max_share / (1 - max_share) * n_other)
    major_idx = df.index[is_major]
    if len(major_idx) > keep:
        rng = np.random.default_rng(seed)
        drop = rng.choice(major_idx, size=len(major_idx) - keep, replace=False)
        mask.loc[drop] = False
    return mask


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()
    acc = {k: [] for k in ("s_p", "s_t", "e_p", "e_t", "i_p", "i_t")}
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        sm, em, im = batch["sentiment_mask"], batch["event_mask"], batch["impact_mask"]
        acc["s_p"] += out["sentiment"].cpu()[sm].tolist()
        acc["s_t"] += batch["sentiment"][sm].tolist()
        acc["e_p"] += out["event_logits"].argmax(-1).cpu()[em].tolist()
        acc["e_t"] += batch["event"][em].tolist()
        acc["i_p"] += out["impact_mean"].cpu()[im].tolist()
        acc["i_t"] += batch["impact"][im].tolist()
    m = {
        "sentiment_pearson": float(pearsonr(acc["s_t"], acc["s_p"])[0]) if len(acc["s_t"]) > 2 else 0.0,
        "event_macro_f1": float(f1_score(acc["e_t"], acc["e_p"], average="macro")) if acc["e_t"] else 0.0,
        "impact_spearman": float(spearmanr(acc["i_t"], acc["i_p"])[0]) if len(acc["i_t"]) > 2 else 0.0,
    }
    m["composite"] = float(np.nanmean(list(m.values())))
    return m


def train(epochs: int = None, limit: int = None):
    with open("config/engine.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    seed = cfg["engine"]["random_seed"]
    set_seed(seed)
    epochs = epochs or mcfg["epochs"]
    device = resolve_device(cfg["engine"]["device"])

    with open("config/taxonomy.yaml", "r", encoding="utf-8") as f:
        labels = [c["name"] for c in yaml.safe_load(f)["classes"]]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    df = pd.read_csv(DATASET)
    if limit:
        df = pd.concat([g.sample(min(len(g), limit), random_state=seed) for _, g in df.groupby("split")])
    df = df.reset_index(drop=True)
    df["event_mask"] = df["has_event_label"].astype(bool)
    train_mask = df["split"] == "train"
    df.loc[train_mask, "event_mask"] = cap_majority_class(df[train_mask], "Other/None", 0.4, seed)

    tr, va = df[train_mask].reset_index(drop=True), df[df["split"] == "val"].reset_index(drop=True)
    counts = tr.loc[tr["event_mask"], "event_type"].value_counts()
    weights = torch.tensor([1.0 / np.sqrt(counts.get(l, 1) + 1) for l in labels], dtype=torch.float32)
    weights = (weights / weights.mean()).to(device)
    print(f"Device={device} train={len(tr)} val={len(va)}")
    print("Event supervision (train, after capping):", counts.to_dict())

    tokenizer = AutoTokenizer.from_pretrained(mcfg["encoder"])
    tr_ds, va_ds = RiskDataset(tr, label_to_idx), RiskDataset(va, label_to_idx)
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr_ds, batch_size=mcfg["batch_size"], shuffle=True, generator=g,
                           collate_fn=make_collate(tr_ds, tokenizer, mcfg["max_length"]))
    va_loader = DataLoader(va_ds, batch_size=128, collate_fn=make_collate(va_ds, tokenizer, mcfg["max_length"]))

    model = MultiTaskRiskModel(mcfg["encoder"], len(labels)).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": mcfg["learning_rate"]},
        {"params": head_params, "lr": mcfg["learning_rate"] * 10},
    ], weight_decay=0.01)
    total_steps = epochs * len(tr_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    out_dir = mcfg["checkpoint_dir"]
    best, history, patience = -np.inf, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), {"total": 0.0, "sentiment": 0.0, "event": 0.0, "impact": 0.0}
        for step, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            losses = masked_multitask_loss(out, batch, weights)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for k in running:
                running[k] += losses[k].item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(tr_loader)} loss {running['total'] / step:.4f}", flush=True)

        val = evaluate(model, va_loader, device, labels)
        record = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
                  **{f"train_loss_{k}": round(v / len(tr_loader), 4) for k, v in running.items()},
                  **{f"val_{k}": round(v, 4) for k, v in val.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)

        if val["composite"] > best:
            best, patience = val["composite"], 0
            os.makedirs(out_dir, exist_ok=True)
            model.encoder.save_pretrained(os.path.join(out_dir, "encoder"))
            tokenizer.save_pretrained(os.path.join(out_dir, "encoder"))
            heads = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            torch.save(heads, os.path.join(out_dir, "heads.pt"))
            with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "base_encoder": mcfg["encoder"], "event_labels": labels,
                    "max_length": mcfg["max_length"], "best_epoch": epoch,
                    "model_version": f"multitask-{mcfg['encoder']}-{cfg['engine']['version']}",
                    "val_metrics": val,
                }, f, indent=2)
            print(f"  saved checkpoint (val composite {best:.4f})")
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping.")
                break

    os.makedirs("reports", exist_ok=True)
    with open("reports/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="rows per split, for smoke tests")
    args = parser.parse_args()
    train(args.epochs, args.limit)
