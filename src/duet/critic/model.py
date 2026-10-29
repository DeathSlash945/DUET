"""Critic architecture: encoder-based identification classifier (Component A).

A standard encoder with a multiple-choice head: one scalar score per
(response, candidate) pair, softmax over the candidate set. The reward used by
preference optimization is the probability mass on the TRUE profile
(probs[..., true_idx]) — see rewards.py.

Default backbone: microsoft/deberta-v3-base (strong on natural-language
inference-style discrimination); distilbert-base-uncased is the fast/low-VRAM
alternative (configs/critic.yaml). The critic is small and frozen w.r.t. the
policy — it never needs the policy's hidden states, so it can run standalone
during pair building and refreshes.
"""

import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from .negatives import NO_MATCH_TEXT, render_profile


def load_critic(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(name_or_path)
    return model, tok


def encode_examples(tok, examples, max_length, device):
    """Tokenize a batch of examples into (B, C, L) tensors + labels (B,)."""
    flat_a, flat_b, n_choices = [], [], len(examples[0]["candidates"])
    for ex in examples:
        for cand in ex["candidates"]:
            flat_a.append(ex["response"])
            flat_b.append(cand)
    enc = tok(flat_a, flat_b, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt")
    b = len(examples)
    input_ids = enc["input_ids"].view(b, n_choices, -1).to(device)
    attention_mask = enc["attention_mask"].view(b, n_choices, -1).to(device)
    labels = torch.tensor([ex["label"] for ex in examples], device=device)
    return input_ids, attention_mask, labels


@torch.no_grad()
def score_examples(model, tok, examples, device, batch_size=16, max_length=256,
                   bf16=True):
    """Softmax probabilities over the candidate set for every example."""
    model.eval()
    probs = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        input_ids, attention_mask, _ = encode_examples(tok, batch, max_length, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.startswith("cuda")):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(probs)  # (N, C)


def p_true(probs, labels) -> torch.Tensor:
    """Reward per example: probability mass on the true profile."""
    return probs.gather(1, labels.view(-1, 1)).squeeze(1)
