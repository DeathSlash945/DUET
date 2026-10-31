"""PAL DPO samples -> our §5 pair contract (the entailment arm's pair builder).

    python -m duet.baselines.pal.to_pairs \
        --samples outputs/pal_runs/pal-gpt2-dpo-samples.json \
        --records data/processed/valid.jsonl \
        [--config configs/dpo_entailment.yaml] [--out PATH] [--limit N]

Stage 02c (upstream decode.py, select-then-generate) emits beam-search
rejections in PAL's OWN prompt format: [{prompt, golden_response,
generated_response}] appended in VALID-split dataloader order. Our
head-to-head trains dpo_entailment through run_dpo, which wants
{prompt, chosen, rejected} rendered in OUR format_prompt — so:

  1. join samples[i] to records[i] POSITIONALLY (eval loaders don't
     shuffle; decode.py appends in iteration order),
  2. validate the join: normalized(golden_response) must equal
     normalized(record.target) for EVERY row — abort loudly rather than
     ever train the entailment arm on misaligned pairs,
  3. re-render the prompt with our format_prompt under the config's
     profile scope, so dpo_entailment vs dpo_critic differ by SIGNAL only,
  4. write data/dpo/dpo_entailment.jsonl = run_dpo's default path for
     run_name: dpo_entailment.

Empty generations are dropped and counted (a beam-search dead end is not a
usable rejection).
"""

import argparse
import json
from pathlib import Path

from duet.training.common import (apply_profile_scope, cfg_get, load_yaml,
                                  read_jsonl, to_ns, write_jsonl)
from duet.training.prompting import format_prompt


def _norm(text):
    return " ".join(text.split())


def join_samples(samples, records):
    """Positional join validated by golden==target. [(record, generation)]."""
    if len(samples) != len(records):
        raise SystemExit(
            f"length mismatch: {len(samples)} samples vs {len(records)} records "
            "— 02c must decode the SAME valid split data/processed/valid.jsonl "
            "was built from")
    bad = [i for i, (s, r) in enumerate(zip(samples, records))
           if _norm(s["golden_response"]) != _norm(r["target"])]
    if bad:
        raise SystemExit(
            f"golden/target mismatch at {len(bad)} position(s) (first: {bad[:5]}) "
            "— refusing to build misaligned pairs")
    return list(zip(records, (s["generated_response"] for s in samples)))


def build_pairs(joined, both=True):
    """DPO rows in the §5 contract; empty rejections dropped (counted)."""
    rows, dropped = [], 0
    for r, gen in joined:
        if not gen or not gen.strip():
            dropped += 1
            continue
        rows.append({
            "prompt": format_prompt(r["history"], r["profile_self"],
                                    r["profile_partner"] if both else []),
            "chosen": r["target"],
            "rejected": gen.strip(),
            "signal": "entailment",
            "dialog_id": r["id"],
        })
    return rows, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples",
                    default="outputs/pal_runs/pal-gpt2-dpo-samples.json")
    ap.add_argument("--records", default="data/processed/valid.jsonl")
    ap.add_argument("--config", default="configs/dpo_entailment.yaml",
                    help="source of the profile scope (both_profiles)")
    ap.add_argument("--out", default=None,
                    help="default: data/dpo/<run_name>.jsonl from --config")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = to_ns(load_yaml(a.config))
    both = cfg_get(cfg.data, "both_profiles", True)
    samples = json.loads(Path(a.samples).read_text(encoding="utf8"))
    records = apply_profile_scope(read_jsonl(a.records), both)
    if a.limit:
        samples, records = samples[:a.limit], records[:a.limit]

    joined = join_samples(samples, records)
    rows, dropped = build_pairs(joined, both=both)
    if not rows:
        raise SystemExit("all rejections were empty — nothing to train on")

    out = a.out or f"data/dpo/{cfg.run_name}.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} entailment pairs -> {out} "
          f"(dropped {dropped} empty rejections, both_profiles={both})")


if __name__ == "__main__":
    main()
