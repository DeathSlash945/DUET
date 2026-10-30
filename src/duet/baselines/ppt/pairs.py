"""PPT pair builder — pseudo-preference data from persona-swapped negatives.

    python -m duet.baselines.ppt.pairs --config configs/ppt.yaml \
        [--limit N] [--model outputs/sft/LATEST] [--device cuda] [--dry-run]

Steps (paper §3):
  1. swap personas — for every §5 record, sample ONE persona from a different
     dialogue (own excluded, deterministic under cfg.train.seed);
  2. pseudo-negative — y' ~ pi_sft(y' | p_swapped, x): generate with the SFT
     policy on the SWAPPED prompt (same history);
  3. pairs — {prompt: TRUE persona + history, chosen: gold, rejected: y'}
     -> data/dpo/ppt.jsonl = run_dpo's default path for run_name: ppt.

--dry-run skips model loading (rejected = placeholder) so the pair plumbing
runs on CPU; tests inject their own generate_fn instead. The generation config
keys live under `generation:` in configs/ppt.yaml (paper leaves sampling
params unstated — vLLM defaults used, flagged in the report).
"""

import argparse
import random

from ...training.common import (apply_profile_scope, cfg_get, load_yaml,
                                read_jsonl, to_ns, write_jsonl)
from ...training.prompting import format_prompt


def sample_swapped_profile(record, pool, rng):
    """One persona from a DIFFERENT dialogue (paper: p_j, j != i).

    `pool` = [(pid, profile)]; candidates whose profile matches the record's
    own (sentence-tuple equality, so shared-persona dialogues are excluded
    too) are never chosen.
    """
    own = tuple(record["profile_self"])
    eligible = [prof for pid, prof in pool if tuple(prof) != own]
    if not eligible:
        raise ValueError("swap pool has no foreign persona")
    return rng.choice(eligible)


def build_swapped_prompts(records, pool, cfg, seed=None):
    """(records, swapped_prompts): one negative-prompt per record."""
    rng = random.Random(seed if seed is not None else cfg.train.seed)
    both = cfg_get(cfg.data, "both_profiles", True)
    records = apply_profile_scope(records, both)
    swapped_profiles = [sample_swapped_profile(r, pool, rng) for r in records]
    prompts = [
        format_prompt(r["history"], sp, r.get("profile_partner"))
        for r, sp in zip(records, swapped_profiles)
    ]
    return records, prompts


def assemble_pairs(records, negatives, meta=None):
    """DPO rows: prompt carries the TRUE persona; rejected was generated
    under the swapped one."""
    both = meta.get("both_profiles", True) if meta else True
    rows = []
    for r, neg in zip(records, negatives):
        rows.append({
            "prompt": format_prompt(r["history"], r["profile_self"],
                                    r.get("profile_partner") if both else []),
            "chosen": r["target"],
            "rejected": neg,
            "signal": "pseudo_preference",
            "dialog_id": r["id"],
        })
    return rows


def generate_negatives(prompts, cfg, model_path, device):
    """y' ~ pi_sft(·|p_swapped, x) — batched HF generate (paper uses vLLM)."""
    import torch
    from ...training.common import load_policy

    cfg.init_from = model_path
    model, tok = load_policy(cfg, trainable=False)
    model.to(device).eval()
    gen = cfg.generation
    out = []
    for start in range(0, len(prompts), gen.batch_size):
        batch = prompts[start:start + gen.batch_size]
        ids = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=cfg.data.max_seq_len).to(device)
        with torch.no_grad():
            y = model.generate(
                **ids, max_new_tokens=gen.max_new_tokens,
                do_sample=gen.do_sample,
                temperature=gen.temperature if gen.do_sample else None,
                top_p=gen.top_p if gen.do_sample else None,
                pad_token_id=tok.pad_token_id,
            )
        for i, seq in enumerate(y):
            text = tok.decode(seq[ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            out.append(text.strip())
        print(f"negatives: {min(start + gen.batch_size, len(prompts))}"
              f"/{len(prompts)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ppt.yaml")
    ap.add_argument("--model", default=None,
                    help="pi_sft for negatives (default: cfg init_from)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip generation (CPU plumbing check)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = to_ns(load_yaml(a.config))
    records = read_jsonl(cfg.data.train_path)
    if a.limit:
        records = records[:a.limit]
    pool = [(r["id"], r["profile_self"]) for r in read_jsonl(cfg.data.train_path)]

    records, swap_prompts = build_swapped_prompts(records, pool, cfg)
    if a.dry_run:
        negatives = ["<NEGATIVE>"] * len(records)
    else:
        negatives = generate_negatives(
            swap_prompts, cfg, a.model or cfg.init_from, a.device)

    rows = assemble_pairs(records, negatives,
                          {"both_profiles": cfg_get(cfg.data, "both_profiles", True)})
    out = a.out or f"data/dpo/{cfg.run_name}.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} PPT pairs -> {out}")
    empties = sum(1 for r in rows if not r["rejected"].strip())
    if empties:
        print(f"warning: {empties} empty negatives (rejected) — DPO will "
              f"reject-side crash or no-op; consider regenerating")


if __name__ == "__main__":
    main()
