"""Critic training + isolation validation (Component A, plan §6 Phase-1 gate).

    python -m duet.critic.train --config configs/critic.yaml \
        [--max-records N] [--epochs N] [--device cpu|cuda]

Pipeline: contract JSONL -> negatives.build_examples -> (response, candidates,
label) with categories real/copy/generic -> plain torch loop (AdamW + linear
warmup) -> validation report with overall AND per-category accuracy -> gate
check (configs critic.gate).

GATE before any critic-guided DPO: beats chance by a wide margin on held-out
data AND the copy/generic probes resolve to the no-match slot. Output:
checkpoints/critic/<run>/ + LATEST symlink + validation_report.json.

Held-out note: the Phase-0 contract has no persona ids, so validation holds out
DIALOGUES with distractors drawn from the train pool. When Kanishk's pipeline
adds persona ids, switch to profile-level holdout (the leakage-proof option).
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .model import encode_examples, load_critic, score_examples
from .negatives import build_example, build_probes
from ..training.common import read_jsonl, save_with_latest, to_ns, load_yaml, cfg_get


def build_examples(records, pool, cfg, rng, max_records=None) -> list:
    """real + copy + generic examples per record (class-balanced)."""
    examples = []
    for rec in (records[:max_records] if max_records else records):
        examples.append(build_example(rec, pool, cfg.candidates, rng))
        examples.extend(build_probes(rec, pool, cfg.candidates, rng))
    return examples


def evaluate(model, tok, examples, cfg, device) -> dict:
    probs = score_examples(model, tok, examples, device,
                           batch_size=cfg.train.eval_batch_size,
                           max_length=cfg.data.max_seq_len,
                           bf16=cfg.train.bf16)
    preds = probs.argmax(dim=1)
    labels = torch.tensor([ex["label"] for ex in examples])

    def acc(sel):
        idx = [i for i, ex in enumerate(examples) if sel(ex)]
        return float((preds[idx] == labels[idx]).float().mean()) if idx else None

    per_cat = {c: acc(lambda ex, c=c: ex["category"] == c)
               for c in ("real", "copy", "generic")}
    n_choices = len(examples[0]["candidates"])
    report = {
        "n": len(examples),
        "chance": 1.0 / n_choices,
        "overall_acc": float((preds == labels).float().mean()),
        "real_acc": per_cat["real"],
        "copy_to_slot_acc": per_cat["copy"],     # copies -> no-match slot
        "generic_to_slot_acc": per_cat["generic"],
        "mean_p_true_real": float(probs[[i for i, e in enumerate(examples)
                                         if e["category"] == "real"]]
                                  .gather(1, labels[[i for i, e in enumerate(examples)
                                                     if e["category"] == "real"]].view(-1, 1)).mean()),
    }
    g = cfg.gate
    report["gate"] = {
        "real_acc_min": g.real_acc_min,
        "copy_to_slot_min": g.copy_to_slot_min,
        "generic_to_slot_min": g.generic_to_slot_min,
        "passed": bool(report["real_acc"] >= g.real_acc_min
                       and report["copy_to_slot_acc"] >= g.copy_to_slot_min
                       and report["generic_to_slot_acc"] >= g.generic_to_slot_min),
    }
    return report


def train_from(model, tok, train_examples, eval_examples, cfg, out_dir,
               device) -> dict:
    """Train an ALREADY-LOADed critic (base or warm-started) and save it.

    Shared by train_critic (from base) and refresh.refresh_critic (from the
    previous critic).
    """
    torch.manual_seed(cfg.train.seed)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    loader = DataLoader(train_examples, batch_size=cfg.train.batch_size,
                        shuffle=True, collate_fn=lambda b: b)
    steps_per_epoch = len(loader)
    total = steps_per_epoch * cfg.train.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total * cfg.train.warmup_ratio), total)

    bf16 = cfg.train.bf16 and device.startswith("cuda")
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in loader:
            input_ids, attention_mask, labels = encode_examples(
                tok, batch, cfg.data.max_seq_len, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step(); sched.step(); opt.zero_grad()
        report = evaluate(model, tok, eval_examples, cfg, device)
        print(f"epoch {epoch + 1}: overall={report['overall_acc']:.3f} "
              f"real={report['real_acc']:.3f} copy->slot={report['copy_to_slot_acc']:.3f} "
              f"generic->slot={report['generic_to_slot_acc']:.3f} "
              f"(chance={report['chance']:.3f})")

    out_dir = Path(out_dir)
    save_with_latest(model, out_dir / "final")
    tok.save_pretrained(out_dir / "final")
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"gate: {'PASSED' if report['gate']['passed'] else 'FAILED'} -> {report['gate']}")
    return report


def train_critic(train_examples, eval_examples, cfg, out_dir, device) -> dict:
    random.seed(cfg.train.seed)
    model, tok = load_critic(cfg.base_model)
    return train_from(model, tok, train_examples, eval_examples, cfg, out_dir,
                      device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/critic.yaml")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = to_ns(load_yaml(a.config))
    if a.epochs:
        cfg.train.epochs = a.epochs
    train_recs = read_jsonl(cfg.data.train_path)
    pool = [(r["id"], r["profile_self"]) for r in train_recs]
    if cfg_get(cfg.candidates, "distractor_pool", None):
        # Fixed-size distractor pool (deterministic subsample) — keeps hard
        # negative search cheap and stable across runs.
        pool = random.Random(cfg.train.seed).sample(
            pool, min(cfg.candidates.distractor_pool, len(pool)))
    train_ex = build_examples(train_recs, pool, cfg, random.Random(cfg.train.seed),
                              a.max_records)
    eval_ex = build_examples(read_jsonl(cfg.data.eval_path), pool, cfg,
                             random.Random(cfg.train.seed + 1), a.max_records)
    print(f"train examples: {len(train_ex)} | eval examples: {len(eval_ex)} | "
          f"device: {a.device}")
    report = train_critic(train_ex, eval_ex, cfg,
                          f"checkpoints/critic/{cfg.run_name}", a.device)
    # Top-level hand-off symlink: checkpoints/critic/LATEST -> <run>/final
    root = Path("checkpoints/critic")
    top = root / "LATEST"
    if top.is_symlink() or top.exists():
        top.unlink()
    top.symlink_to(Path(cfg.run_name) / "final")
    raise SystemExit(0 if report["gate"]["passed"] else 2)


if __name__ == "__main__":
    main()
