"""Cross-encoder NLI scoring — the entailment signal (PAL-style) used by the
entailment DPO arm and by the contradiction metric of the evaluation harness.

The model is a standard sentence-pair cross-encoder (default:
``cross-encoder/nli-roberta-base``, SNLI+MNLI). ``score_pairs`` returns the
softmax probabilities of the three classes with labels looked up from the
model config, so the entailment/contradiction indices are never assumed.

Deliberately separate from the identification critic: the entailment arm and
the critic arm differ in *signal only*, and the harness must not reuse the
selection model for grading (the report's two-seed separation rule).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "cross-encoder/nli-roberta-base"


def load_nli(name: str = DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    labels = {i: str(l).lower() for i, l in
              (model.config.id2label or {}).items()}
    if not labels:  # some checkpoints omit id2label
        labels = {0: "contradiction", 1: "entailment", 2: "neutral"}
    model._nli_labels = labels   # consumed by score_pairs
    return model, tok, labels


@torch.no_grad()
def score_pairs(model, tok, pairs, device="cuda", batch_size=32,
                max_length=256) -> list:
    """Score (premise, hypothesis) pairs -> list of label->prob dicts."""
    labels = getattr(model, "_nli_labels",
                     {0: "contradiction", 1: "entailment", 2: "neutral"})
    model.to(device)
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk],
                  padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        logits = model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu()
        for row in probs:
            out.append({labels[j]: float(v) for j, v in enumerate(row)})
    return out


def scrub(probs: dict, key: str) -> float:
    """Fetch a class probability with a clear failure if mislabelled."""
    if key in probs:
        return probs[key]
    raise KeyError(f"NLI model has no '{key}' label (got {sorted(probs)})")
