"""Stage 03 — preference optimization (TRL DPO). Signal-agnostic.

The pairs come from the builder matching the config's signal type
(entailment = PAL/PPT pair builders, identification = critic/rewards.py);
the trainer itself only ever sees {prompt, chosen, rejected} rows.

    python -m duet.training.run_dpo --config configs/dpo_critic.yaml \
        [--pairs PATH] [--max-examples N] [--dump-generations N]

Pairs default path: data/dpo/<run_name>.jsonl
"""

import argparse
from pathlib import Path

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from .common import (config_args, dump_generations, load_policy, load_yaml,
                     read_jsonl, save_with_latest, to_ns)

REQUIRED = ("prompt", "chosen", "rejected")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", default=None,
                    help="default: data/dpo/<run_name>.jsonl")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--dump-generations", type=int, default=0)
    a = ap.parse_args()

    cfg = to_ns(load_yaml(a.config))
    pairs_path = a.pairs or f"data/dpo/{cfg.run_name}.jsonl"
    pairs = read_jsonl(pairs_path)
    if a.max_examples:
        pairs = pairs[:a.max_examples]
    missing = [k for k in REQUIRED if k not in pairs[0]]
    if missing:
        raise SystemExit(f"{pairs_path}: rows missing {missing} — pairs must be "
                         "{prompt, chosen, rejected} (project_plan.md §5)")
    print(f"pairs: {len(pairs)} from {pairs_path}")

    model, tok = load_policy(cfg, trainable=True)  # init_from -> continue SFT adapter

    args = DPOConfig(
        output_dir=f"outputs/{cfg.run_name}",
        max_length=cfg.data.max_seq_len,
        **{**config_args(vars(cfg.train), DPOConfig),
           "beta": cfg.train.beta,
           "eval_strategy": "no",          # real eval = generation dump + harness
           "save_strategy": "no",
           "report_to": [],
           "logging_steps": 5},
    )
    ds = Dataset.from_list(pairs)
    trainer = DPOTrainer(model=model, ref_model=None,  # implicit ref: adapter off
                         args=args, train_dataset=ds, processing_class=tok)
    trainer.train()

    save_with_latest(trainer.model, Path(args.output_dir) / "final")
    print(f"saved adapter -> {args.output_dir}/final (LATEST symlink updated)")

    if a.dump_generations:
        eval_recs = read_jsonl(cfg.data.eval_path) if Path(cfg.data.eval_path).exists() else []
        n = dump_generations(
            trainer.model, tok, eval_recs,
            Path(args.output_dir) / "generations.jsonl",
            run_id=cfg.run_name, model_name=cfg.base_model,
            condition=cfg.run_name, limit=a.dump_generations,
        )
        print(f"wrote {n} generations -> {args.output_dir}/generations.jsonl")


if __name__ == "__main__":
    main()
