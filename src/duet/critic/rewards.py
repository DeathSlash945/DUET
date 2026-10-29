"""Identification signal -> preference pairs (plan §6 item: rewards.py).

Reward definition (dpo_critic.yaml `signal.score`): for a policy response y
given true profile p and candidate set C = {p, d1..dk, NO_MATCH}:

    r(y) = P_critic(label = p | y, C)          # softmax prob on the true slot

Preference pair construction (matched with the entailment arm's recipe so the
head-to-head is signal-only): chosen = the GOLD response (traceable to p by
construction — the conservative anchor from the old prototype), rejected = the
sampled policy response with the LOWEST r(y). Sampling temperature and count
come from the caller (training loop integration, Phase 1).

    python -m duet.critic.rewards --config configs/critic.yaml \
        --generations outputs/sft/generations.jsonl \
        --out data/dpo/dpo_critic.jsonl [--device cuda]

Input generations.jsonl: the §5 dump contract (one row per sampled response,
`dialog_id` joins back to the contract record). Output rows: §5 DPO pair
contract {prompt, chosen, rejected} + signal metadata for auditing.
"""

import argparse
import random
import torch

from .model import load_critic, score_examples, p_true
from .negatives import build_example, render_profile, NO_MATCH_TEXT
from ..training.common import read_jsonl, write_jsonl
from ..training.prompting import format_prompt


def score_response(model, tok, response, record, cfg, rng, device) -> float:
    """r(y) = P_critic identifies the true profile for one response."""
    ex = build_example(record, cfg.pool, cfg.candidates, rng)
    ex["response"] = response
    probs = score_examples(model, tok, [ex], device,
                           max_length=cfg.data.max_seq_len,
                           bf16=cfg.train.bf16)
    return float(p_true(probs, torch.tensor([ex["label"]]))[0])


def build_pairs(records_by_id, generations, cfg, model, tok, device,
                seed=42) -> list:
    """Group sampled responses by dialog, pair gold-vs-worst-identified."""
    rng = random.Random(seed)
    by_dialog = {}
    for row in generations:
        by_dialog.setdefault(row["dialog_id"], []).append(row["response"])

    pairs = []
    for dialog_id, responses in sorted(by_dialog.items()):
        record = records_by_id[dialog_id]
        scores = [score_response(model, tok, r, record, cfg, rng, device)
                  for r in responses]
        worst = min(range(len(responses)), key=lambda i: scores[i])
        best = max(range(len(responses)), key=lambda i: scores[i])
        prompt = format_prompt(record["history"], record["profile_self"],
                               record.get("profile_partner"))
        pairs.append({
            "prompt": prompt,
            "chosen": record["target"],          # gold anchor (conservative)
            "rejected": responses[worst],        # least identifiable response
            "signal": "identification",
            "score_chosen": score_response(model, tok, record["target"],
                                           record, cfg, rng, device),
            "score_rejected": scores[worst],
            "score_best_sample": scores[best],
            "n_samples": len(responses),
            "dialog_id": dialog_id,
        })
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/critic.yaml")
    ap.add_argument("--critic", default=None,
                    help="critic ckpt dir (default: cfg checkpoint_dir/LATEST)")
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", default="data/dpo/dpo_critic.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from ..training.common import load_yaml, to_ns, cfg_get
    cfg = to_ns(load_yaml(a.config))
    ckpt = a.critic or f"{cfg_get(cfg, 'checkpoint_dir', 'checkpoints/critic')}/LATEST"
    model, tok = load_critic(ckpt)
    model.to(a.device)

    train_recs = read_jsonl(cfg.data.train_path)
    cfg.pool = [(r["id"], r["profile_self"]) for r in train_recs]
    records_by_id = {r["id"]: r for r in train_recs}

    pairs = build_pairs(records_by_id, read_jsonl(a.generations), cfg,
                        model, tok, a.device, seed=cfg.train.seed)
    write_jsonl(a.out, pairs)
    gaps = [p["score_chosen"] - p["score_rejected"] for p in pairs]
    print(f"wrote {len(pairs)} pairs -> {a.out}")
    print(f"mean score gap (chosen - rejected): {sum(gaps)/len(gaps):.3f}")
    print(f"pairs where rejected out-scores gold: "
          f"{sum(1 for g in gaps if g < 0)}/{len(pairs)}")


if __name__ == "__main__":
    main()
