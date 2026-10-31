"""Evaluation harness — Table I metrics from §5 generation dumps.

Implements the six pre-registered columns so the table can be filled from the
contract alone, without touching any training or selection code:

    identifiability   critic accuracy on the generated response against a
                      FIXED 5-way candidate set (1 true + 3 distractors +
                      no-match), per dialog; chance 0.20
    copy              n-gram overlap between response and profile
    hallucination     fraction of responses with a first-person persona claim
                      that has no content-word support in the profile
    contradiction     NLI contradiction rate (profile premise -> response)
    partner recall    fraction of partner facts the response references
                      (undefined for self-only conditions: the contract never
                      shows the partner section)
    drift             persona-mention drop from shallow (<=2 turns) to deep
                      (>=6 turns) dialogues, in percentage points

Candidate sets are seeded by dialog id, so a dialogue gets the *same*
distractors under every condition — conditions stay comparable.

    python -m duet.eval.harness --dumps reports/dumps_256/*.jsonl \
        --out reports/harness_table.md
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

import torch

from .nli import load_nli, score_pairs, scrub
from ..training.common import read_jsonl

STOP = set(
    "a an the i you he she it we they me my mine your our their them to of in "
    "on at for with and or but is am are was were be been being have has had "
    "do does did this that these those there here as if then than so very "
    "like love enjoy also just really".split()
)


def content_words(text: str) -> list[str]:
    out = []
    for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        if tok not in STOP and len(tok) > 1:
            out.append(tok)
    return out


def ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def copy_rate(response: str, profile_sents: list[str], n: int) -> float:
    """Share of response n-grams that also occur in the profile."""
    prof = ngrams(content_words(" ".join(profile_sents)), n)
    resp = ngrams(content_words(response), n)
    if not resp or not prof:
        return 0.0
    return len(resp & prof) / len(resp)


def unsupported_claim(response: str, profile_sents: list[str]) -> bool:
    """True if a first-person persona claim lacks profile support."""
    prof_words = set(content_words(" ".join(profile_sents)))
    for sent in response.replace("!", ".").replace("?", ".").split("."):
        s = sent.strip().lower()
        if not (s.startswith("i ") or " i " in s):
            continue
        words = content_words(s)
        if words and not (set(words) & prof_words):
            return True
    return False


def partner_recall(response: str, partner_sents: list[str]) -> float | None:
    """Fraction of partner facts referenced by the response."""
    if not partner_sents:
        return None
    resp_words = set(content_words(response))
    hits = sum(1 for s in partner_sents
               if set(content_words(s)) & resp_words)
    return hits / len(partner_sents)


def self_mention(response: str, profile_sents: list[str]) -> float:
    resp_words = set(content_words(response))
    if not profile_sents:
        return 0.0
    return sum(1 for s in profile_sents
               if set(content_words(s)) & resp_words) / len(profile_sents)


def is_self_only(condition: str) -> bool:
    return "selfonly" in condition or "self_only" in condition


def score_condition(rows, records_by_id, critic_bundle, nli_bundle,
                    device, limit=None) -> dict:
    from ..critic.negatives import build_example
    from ..critic.model import p_true, score_examples
    from ..training.common import cfg_get, load_yaml, to_ns

    rows = rows[:limit] if limit else rows
    critic_cfg, critic_model, critic_tok = critic_bundle
    cfg = critic_cfg

    train_recs = read_jsonl(cfg.data.train_path)
    pool_rng = random.Random(cfg_get(cfg.train, "seed", 42))
    pool = [(r["id"], r["profile_self"]) for r in train_recs]
    if cfg_get(cfg.candidates, "distractor_pool", None):
        pool = pool_rng.sample(pool, min(cfg.candidates.distractor_pool,
                                         len(pool)))
    by_id = {r["id"]: r for r in train_recs}

    examples, labels = [], []
    for row in rows:
        rec = records_by_id.get(row["dialog_id"]) or by_id.get(row["dialog_id"])
        if rec is None:
            continue
        rng = random.Random(zlib.crc32(str(row["dialog_id"]).encode()))
        ex = build_example(rec, pool, cfg.candidates, rng)
        ex["response"] = row["response"]
        examples.append(ex)
        labels.append(ex["label"])

    probs = score_examples(critic_model, critic_tok, examples, device,
                           batch_size=cfg_get(cfg.train, "eval_batch_size", 8),
                           max_length=cfg.data.max_seq_len,
                           bf16=cfg_get(cfg.train, "bf16", True))
    preds = probs.argmax(dim=1)
    lab = torch.tensor(labels)
    ident_acc = float((preds == lab).float().mean()) if len(labels) else 0.0
    ptrue = p_true(probs, lab).tolist() if len(labels) else []

    nli_model, nli_tok, _ = nli_bundle

    premise_pairs = [("\n".join(records_by_id[row["dialog_id"]]["profile_self"]),
                      row["response"])
                     for row in rows if row["dialog_id"] in records_by_id]
    nli_scores = score_pairs(nli_model, nli_tok, premise_pairs,
                             device=device, batch_size=32)
    contra = [scrub(s, "contradiction") for s in nli_scores]
    contra_rate = (sum(1 for c in contra if c > 0.5) / len(contra)) if contra else 0.0

    copies1, copies3, halluc, partner, selfm, lengths, depths = [], [], [], [], [], [], []
    self_only = is_self_only(rows[0].get("condition", ""))
    for row in rows:
        prof = row.get("profile_self") or []
        partner_prof = row.get("profile_partner") or []
        resp = row.get("response", "")
        copies1.append(copy_rate(resp, prof, 1))
        copies3.append(copy_rate(resp, prof, 3))
        halluc.append(1.0 if unsupported_claim(resp, prof) else 0.0)
        partner.append(partner_recall(resp, partner_prof))
        selfm.append(self_mention(resp, prof))
        lengths.append(len(resp.split()))
        rec = records_by_id.get(row["dialog_id"], {})
        depths.append(len(rec.get("history", [])))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    shallow = [m for m, d in zip(selfm, depths) if d <= 2]
    deep = [m for m, d in zip(selfm, depths) if d >= 6]
    drift = None
    if shallow and deep:
        drift = 100.0 * (mean(shallow) - mean(deep))

    return {
        "condition": rows[0].get("condition", "?"),
        "n": len(rows),
        "identifiability": ident_acc,
        "p_true": mean(ptrue),
        "copy@1": mean(copies1),
        "copy@3": mean(copies3),
        "hallucination": mean(halluc),
        "contradiction": contra_rate,
        "partner_recall": None if self_only else mean(partner),
        "drift_pp": drift,
        "resp_len": mean(lengths),
    }


def fmt(x, digits=3):
    return "---" if x is None else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", default="reports/harness_table.md")
    ap.add_argument("--critic-config", default="configs/critic.yaml")
    ap.add_argument("--critic", default="checkpoints/critic/LATEST")
    ap.add_argument("--nli-model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from ..critic.model import load_critic, score_examples  # noqa: F401
    from ..eval.nli import DEFAULT_MODEL, load_nli
    from ..training.common import load_yaml, to_ns

    critic_model, critic_tok = load_critic(a.critic)
    critic_model.to(a.device)
    critic_cfg = to_ns(load_yaml(a.critic_config))
    nli_model, nli_tok, _ = load_nli(a.nli_model or DEFAULT_MODEL)

    # contract records for joins (history depth, gold target)
    records = {}
    for path in a.dumps:
        for row in read_jsonl(path):
            records.setdefault(row["dialog_id"], None)
    train = {r["id"]: r for r in read_jsonl(critic_cfg.data.train_path)}
    records_by_id = {k: train[k] for k in records if k in train}

    results = []
    for path in a.dumps:
        rows = read_jsonl(path)
        if not rows:
            continue
        rows_all = rows
        results.append(score_condition(
            rows_all, records_by_id,
            (critic_cfg, critic_model, critic_tok),
            (nli_model, nli_tok, None), a.device, limit=a.limit))

    md = ["| condition | n | identif. | copy@1 | copy@3 | halluc. | "
          "contrad. | partner rec. | drift (pp) | len |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(
            f"| {r['condition']} | {r['n']} | {fmt(r['identifiability'])} | "
            f"{fmt(r['copy@1'])} | {fmt(r['copy@3'])} | "
            f"{fmt(r['hallucination'])} | {fmt(r['contradiction'])} | "
            f"{fmt(r['partner_recall'])} | {fmt(r['drift_pp'], 1)} | "
            f"{fmt(r['resp_len'], 1)} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n")

    tex = [r"% Table I rows (eval/harness.py) — insert with \usepackage{booktabs}",
           r"% columns: Identif.$\uparrow$ & Copy$\downarrow$ & Halluc.$\downarrow$ & Contrad.$\downarrow$ & Partner recall$\uparrow$ & Drift$\downarrow$"]
    for r in results:
        tex.append(f"{r['condition']} & {fmt(r['identifiability'])} & "
                   f"{fmt(r['copy@3'])} & {fmt(r['hallucination'])} & "
                   f"{fmt(r['contradiction'])} & "
                   f"{fmt(r['partner_recall'])} & {fmt(r['drift_pp'], 1)} \\\\")
    Path(str(a.out).replace(".md", ".tex")).write_text("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\nwrote {a.out} (+ .tex rows)")


if __name__ == "__main__":
    main()
