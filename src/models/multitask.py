import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils import resolve_device

IMPACT_MIN, IMPACT_MAX = 1.0, 10.0


class MultiTaskRiskModel(nn.Module):
    """
    Shared transformer encoder with three task heads:

      sentiment  tanh regression in [-1, 1]
      event      softmax over the taxonomy classes
      impact     heteroscedastic regression: mean severity in [1, 10] plus log-variance,
                 trained with Gaussian NLL so the model can say *how sure* it is
    """

    def __init__(self, encoder_name: str, n_events: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        def mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, out_dim))

        self.sentiment_head = mlp(1)
        self.event_head = mlp(n_events)
        self.impact_head = mlp(2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = self.dropout((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0))

        impact = self.impact_head(pooled)
        return {
            "sentiment": torch.tanh(self.sentiment_head(pooled).squeeze(-1)),
            "event_logits": self.event_head(pooled),
            "impact_mean": IMPACT_MIN + (IMPACT_MAX - IMPACT_MIN) * torch.sigmoid(impact[:, 0]),
            "impact_log_var": impact[:, 1].clamp(-3.0, 4.0),
        }


def masked_multitask_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    event_class_weights: Optional[torch.Tensor] = None,
    task_weights: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Each row contributes only to the tasks it has labels for (PhraseBank has no impact,
    Benzinga/tweets have no gold sentiment), so the masks are applied before averaging.
    """
    task_weights = task_weights or {"sentiment": 1.0, "event": 1.0, "impact": 0.5}
    zero = out["sentiment"].sum() * 0.0
    losses = {}

    m = batch["sentiment_mask"]
    losses["sentiment"] = (
        F.mse_loss(out["sentiment"][m], batch["sentiment"][m]) if m.any() else zero
    )

    m = batch["event_mask"]
    losses["event"] = (
        F.cross_entropy(out["event_logits"][m], batch["event"][m], weight=event_class_weights)
        if m.any() else zero
    )

    m = batch["impact_mask"]
    if m.any():
        mu, log_var, y = out["impact_mean"][m], out["impact_log_var"][m], batch["impact"][m]
        losses["impact"] = (0.5 * (log_var + (y - mu) ** 2 / log_var.exp())).mean()
    else:
        losses["impact"] = zero

    losses["total"] = sum(task_weights[k] * losses[k] for k in ("sentiment", "event", "impact"))
    return losses


def impact_confidence(sigma: np.ndarray, tolerance: float = 1.5) -> np.ndarray:
    """P(|realised - predicted| <= tolerance) under the model's own Gaussian predictive."""
    return np.array([math.erf(tolerance / (s * math.sqrt(2))) for s in np.atleast_1d(sigma)])


def fit_impact_deciles(raw_predictions: np.ndarray) -> List[float]:
    """Nine knots that split held-out impact predictions into equal-count deciles."""
    return [round(float(q), 5) for q in np.quantile(raw_predictions, np.linspace(0.1, 0.9, 9))]


def to_impact_score(raw: np.ndarray, knots: Optional[List[float]]) -> np.ndarray:
    """
    Maps the regression mean onto the 1-10 decile scale the labels were built on. The Gaussian
    head shrinks toward the centre when news is ambiguous, so without this the score rarely
    leaves 4-7; rank order (and Spearman) is unchanged.
    """
    if not knots:
        return np.clip(np.round(raw), 1, 10).astype(int)
    return (1 + np.searchsorted(np.asarray(knots), raw, side="right")).astype(int)


class RiskModelPredictor:
    """Loads a trained checkpoint and runs batched inference on raw strings."""

    def __init__(self, checkpoint_dir: str = "models/risk_engine", device: str = "auto"):
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.labels: List[str] = self.config["event_labels"]
        self.max_length = self.config["max_length"]
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, "encoder"))
        self.model = MultiTaskRiskModel(os.path.join(checkpoint_dir, "encoder"), len(self.labels))
        state = torch.load(os.path.join(checkpoint_dir, "heads.pt"), map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        self.version = self.config.get("model_version", "multitask-distilroberta")

    @torch.inference_mode()
    def predict(self, texts: List[str], batch_size: int = 64) -> Dict[str, np.ndarray]:
        sent, ev_probs, imp_mu, imp_sigma = [], [], [], []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            out = self.model(enc["input_ids"], enc["attention_mask"])
            sent.append(out["sentiment"].float().cpu().numpy())
            ev_probs.append(torch.softmax(out["event_logits"].float(), -1).cpu().numpy())
            imp_mu.append(out["impact_mean"].float().cpu().numpy())
            imp_sigma.append(torch.exp(0.5 * out["impact_log_var"].float()).cpu().numpy())
        probs = np.vstack(ev_probs) if ev_probs else np.zeros((0, len(self.labels)))
        sigma = np.concatenate(imp_sigma) if imp_sigma else np.zeros(0)
        raw = np.concatenate(imp_mu) if imp_mu else np.zeros(0)
        return {
            "sentiment": np.concatenate(sent) if sent else np.zeros(0),
            "event_probs": probs,
            "event_type": np.array([self.labels[j] for j in probs.argmax(1)]),
            "event_confidence": probs.max(1) if len(probs) else np.zeros(0),
            "impact_raw": raw,
            "impact_score": to_impact_score(raw, self.config.get("impact_deciles")),
            "impact_sigma": sigma,
            "impact_confidence": impact_confidence(sigma) if len(sigma) else np.zeros(0),
        }
