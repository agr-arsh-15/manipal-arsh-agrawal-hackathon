from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_recall_fscore_support


def sentiment_to_label(scores: Sequence[float], threshold: float = 0.33) -> List[str]:
    return ["positive" if s > threshold else ("negative" if s < -threshold else "neutral") for s in scores]


def sentiment_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "pearson_r": round(float(pearsonr(y_true, y_pred)[0]), 4),
        "macro_f1": round(float(f1_score(sentiment_to_label(y_true), sentiment_to_label(y_pred), average="macro")), 4),
        "accuracy": round(float(accuracy_score(sentiment_to_label(y_true), sentiment_to_label(y_pred))), 4),
    }


def event_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = None) -> Dict:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = [i for i, s in enumerate(support) if s > 0]
    return {
        "n": int(len(y_true)),
        # Macro-F1 is averaged over classes that actually occur in y_true so absent classes
        # do not silently drag the score to zero.
        "macro_f1": round(float(np.mean([f[i] for i in present])) if present else 0.0, 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class": {
            lab: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                  "f1": round(float(f[i]), 3), "support": int(support[i])}
            for i, lab in enumerate(labels)
        },
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def impact_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    high_true, high_pred = y_true > 7, y_pred > 7
    tp = int((high_true & high_pred).sum())
    return {
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "spearman_rho": round(float(spearmanr(y_true, y_pred)[0]), 4),
        # Module B triggers on impact > 7, so precision/recall at that threshold is what matters downstream.
        "high_impact_precision": round(tp / max(int(high_pred.sum()), 1), 4),
        "high_impact_recall": round(tp / max(int(high_true.sum()), 1), 4),
        "high_impact_base_rate": round(float(high_true.mean()), 4),
    }
