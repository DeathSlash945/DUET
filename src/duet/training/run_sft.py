"""Stage 02 — supervised fine-tuning of the DUET policy (TRL + PEFT).

    python -m duet.training.run_sft --config configs/sft.yaml \
        [--max-examples N] [--dump-generations N]

Data: §5 contract JSONL (configs/sft.yaml data.* paths). Prompt format comes
from duet.training.prompting — the single source shared with DPO and eval.
"""

import argparse
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from .common import (apply_profile_scope, config_args, dump_generations,
                     cfg_get, load_policy, load_yaml, read_jsonl,
                     save_with_latest, to_ns)
from .prompting import format_sft_text

TEXT_FIELD = "__text"


def build_rows(records, limit=None):
    rows = records[:limit] if limit else records
    return [{**r, TEXT_FIELD: format_sft_text(r)} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="cap training rows (smoke runs)")
    ap.add_argument("--max-eval-examples", type=int, default=1024,
                    help="cap eval rows per eval pass (full eval lives in the harness)")
    ap.add_argument("--dump-generations", type=int, default=0,
                    help="write N greedy generations to outputs/<run>/generations.jsonl")
    a = ap.parse_args()

    cfg = to_ns(load_yaml(a.config))
    both = cfg_get(cfg.data, "both_profiles", True)
    train_recs = apply_profile_scope(read_jsonl(cfg.data.train_path), both)
    eval_recs = read_jsonl(cfg.data.eval_path) if Path(cfg.data.eval_path).exists() else None
    if eval_recs is not None:
        eval_recs = apply_profile_scope(eval_recs, both)[:a.max_eval_examples]
    print(f"train rows: {len(train_recs)} | eval rows: {len(eval_recs) if eval_recs else 0}"
          f" | both_profiles: {both}")

    train_ds = Dataset.from_list(build_rows(train_recs, a.max_examples))
    eval_ds = Dataset.from_list(build_rows(eval_recs)) if eval_recs else None

    model, tok = load_policy(cfg)

    out_dir = f"outputs/{cfg.run_name}"
    args = SFTConfig(
        output_dir=out_dir,
        dataset_text_field=TEXT_FIELD,
        max_length=cfg.data.max_seq_len,
        **{**config_args(vars(cfg.train), SFTConfig),
           "per_device_eval_batch_size": 16,
           "eval_strategy": "epoch" if eval_ds is not None else "no",
           "save_strategy": "no",
           "report_to": [],
           "logging_steps": 5},
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                         eval_dataset=eval_ds, processing_class=tok)
    trainer.train()

    save_with_latest(trainer.model, Path(out_dir) / "final")
    print(f"saved adapter -> {out_dir}/final (LATEST symlink updated)")

    if a.dump_generations:
        path = dump_generations(
            trainer.model, tok, eval_recs or train_recs,
            Path(out_dir) / "generations.jsonl",
            run_id=cfg.run_name, model_name=cfg.base_model,
            condition=cfg.run_name, limit=a.dump_generations,
        )
        print(f"wrote {path} generations -> {out_dir}/generations.jsonl")


if __name__ == "__main__":
    main()
