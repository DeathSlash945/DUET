"""Fixed-interval adversarial refresh (plan §6: refresh.py).

Why: a static critic can be EXPLOITED — the policy climbs the critic's fixed
decision boundary instead of getting more persona-faithful (GAN discriminator
going stale). Fix per plan: retrain the critic at fixed step intervals against
the CURRENT policy's generations, never continuously (cost + stability).

    from duet.critic.refresh import refresh_critic
    refresh_critic(cfg, policy_ckpt="outputs/dpo_critic/LATEST",
                   step=500, prev_critic="checkpoints/critic/LATEST")

Mechanics:
  1. sample fresh policy responses on the training set (§5 dump rows),
  2. rebuild candidate sets with the new responses treated as the "real"
     class (they carry the true profile by conditioning) + fresh copy and
     generic probes,
  3. warm-start from the PREVIOUS critic (not from base — the boundary should
     track the policy, not restart),
  4. save checkpoints/critic/refresh-<step>/ + LATEST + append one line to
     refresh_history.jsonl (step, n_examples, val acc) — the policy-vs-critic
     dynamics log the plan asks for.
"""

import json
import random
from pathlib import Path

import torch

from .model import load_critic
from .negatives import build_example, build_probes
from .train import train_from
from ..training.common import read_jsonl


def build_refresh_examples(records, pool, cfg, rng, policy_responses=None) -> list:
    """Real class = gold targets + current policy generations (both are
    conditioned on the true profile, so both must be identifiable)."""
    examples, probes = [], []
    policy_by_id = {}
    for row in (policy_responses or []):
        policy_by_id.setdefault(row["dialog_id"], []).append(row["response"])
    for rec in records:
        ex = build_example(rec, pool, cfg.candidates, rng)
        examples.append(ex)
        probes.extend(build_probes(rec, pool, cfg.candidates, rng))
        for resp in policy_by_id.get(rec["id"], []):
            examples.append({**ex, "response": resp})
    return examples + probes


def refresh_critic(cfg, policy_responses, step: int, device: str,
                   prev_critic: str = "checkpoints/critic/LATEST",
                   out_root: str = "checkpoints/critic",
                   records=None) -> Path:
    """One refresh round. `policy_responses` = §5 generations rows on TRAIN."""
    out_root = Path(out_root)
    rng = random.Random(cfg.train.seed + step)
    if records is None:
        records = read_jsonl(cfg.data.train_path)
    pool = [(r["id"], r["profile_self"]) for r in records]

    examples = build_refresh_examples(records, pool, cfg, rng, policy_responses)
    eval_ex = build_refresh_examples(
        read_jsonl(cfg.data.eval_path), pool, cfg, rng, None)
    # NOTE: eval set stays GOLD-only + probes across all rounds — a fixed
    # yardstick, so refresh rounds are comparable to the Phase-1 gate.

    model, tok = load_critic(prev_critic)
    model.to(device)
    out_dir = out_root / f"refresh-{step}"
    report = train_from(model, tok, examples, eval_ex, cfg, out_dir, device)

    latest = out_root / "LATEST"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(Path(out_dir.name) / "final")

    with open(out_root / "refresh_history.jsonl", "a") as f:
        f.write(json.dumps({
            "step": step,
            "n_examples": len(examples),
            "val_overall_acc": report["overall_acc"],
            "val_real_acc": report["real_acc"],
            "gate_passed": report["gate"]["passed"],
        }) + "\n")
    return out_dir
