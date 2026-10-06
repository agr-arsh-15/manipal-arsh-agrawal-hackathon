from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import resolve_device

# Natural-language hypotheses per taxonomy class. NLI models score these far better than bare
# class names such as "Other/None".
HYPOTHESES = {
    "Geopolitical": "This text is about geopolitics, war, military conflict, sanctions or trade tariffs.",
    "Macroeconomic": "This text is about the macroeconomy, interest rates, inflation, GDP or central banks.",
    "Credit Event": "This text is about a credit event such as a rating downgrade, debt default or bankruptcy.",
    "Merger/Acquisition": "This text is about a merger, acquisition, takeover or stake purchase.",
    "Product Launch": "This text is about a new product launch, release or regulatory product approval.",
    "Regulatory/Legal": "This text is about a lawsuit, fine, regulation or government investigation.",
    "Earnings/Guidance": "This text is about company earnings, revenue, profit or financial guidance.",
    "Operational/Supply-chain": "This text is about an operational disruption, recall, strike, outage or supply chain problem.",
    "Other/None": "This text is general market commentary, an analyst rating or a stock price move.",
}
LABELS = list(HYPOTHESES.keys())


class ZeroShotEventClassifier:
    """Batched NLI zero-shot classifier over the 9-class event taxonomy."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: str = "auto"):
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
        self.model = self.model.to(self.device).eval()
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entail_idx = label2id.get("entailment", 2)
        self.hypotheses = [HYPOTHESES[l] for l in LABELS]

    @torch.inference_mode()
    def predict_proba(self, texts: List[str], batch_texts: int = 8) -> np.ndarray:
        """Returns an (n_texts, 9) matrix of probabilities over LABELS, in input order."""
        n_h = len(self.hypotheses)
        probs = np.zeros((len(texts), n_h), dtype=np.float32)
        # Length-sorted batches minimise padding, which dominates cost for short headlines.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_texts):
            idx = order[start:start + batch_texts]
            chunk = [texts[i] for i in idx]
            premises = [t for t in chunk for _ in range(n_h)]
            hyps = self.hypotheses * len(chunk)
            enc = self.tokenizer(premises, hyps, truncation="only_first", max_length=96,
                                 padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits[:, self.entail_idx].view(len(chunk), n_h)
            probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        probs = self.predict_proba(texts)
        return [(LABELS[int(p.argmax())], float(p.max())) for p in probs]


def resolve_event_label(rule_label: str, rule_conf: float, zs_label: str, zs_conf: float,
                        zs_threshold: float = 0.5) -> Tuple[str, float, str]:
    """
    Reconciles the keyword rule with the zero-shot model.
    Returns (event_type, confidence, provenance).
    """
    if rule_label == zs_label:
        return zs_label, max(zs_conf, rule_conf), "agree"
    if zs_conf >= zs_threshold:
        return zs_label, zs_conf, "zeroshot"
    if rule_label != "Other/None":
        return rule_label, rule_conf * 0.8, "rule"
    return "Other/None", 1.0 - zs_conf, "default"
