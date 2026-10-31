"""Sample greedy responses from a checkpoint (candidate dumps for pair building).

Writes the §5 generations contract so the same file feeds
``duet.critic.rewards`` (critic arm), the PPT pair builder, and
``duet.eval.prelim``. Unlike the training dump (limit=64), this is meant for
scale: it samples one response per dialog over the contract's train split.

    python -m duet.eval.sample --adapter outputs/sft/final \
        --limit 2000 --out data/dpo/cand_sft.jsonl --condition sft_bidir

The adapter directory comes from any checkpoint saved by the training
wrappers (``outputs/<run>/final``); base model/tokenizer come from
``--config`` (default: configs/sft.yaml).
"""

import argparse

from ..training.common import (cfg_get, dump_generations, load_policy,
                               load_yaml, read_jsonl, to_ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--adapter", required=True,
                    help="checkpoint dir with the LoRA adapter (e.g. outputs/sft/final)")
    ap.add_argument("--data", default=None,
                    help="contract JSONL (default: config data.train_path)")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="sample")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--samples-per-dialog", type=int, default=1,
                    help="K samples per dialog (K>1 = candidate pool for "
                         "select-then-generate arms; forces sampling)")
    ap.add_argument("--do-sample", action="store_true",
                    help="sample instead of greedy (implied by K>1)")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()

    cfg = to_ns(load_yaml(a.config))
    path = a.data or cfg.data.train_path
    records = read_jsonl(path)[a.offset:]

    # load_policy reads init_from to continue an adapter; it only needs
    # base_model + lora/bf16 keys, which every training config carries.
    cfg.init_from = a.adapter
    model, tok = load_policy(cfg, trainable=False)
    if a.device:
        model.to(a.device)

    n = dump_generations(
        model, tok, records, a.out,
        run_id="sample", model_name=cfg.base_model, condition=a.condition,
        max_new_tokens=a.max_new_tokens, limit=a.limit,
        samples_per_dialog=a.samples_per_dialog,
        do_sample=a.do_sample or a.samples_per_dialog > 1,
        temperature=a.temperature,
    )
    print(f"wrote {n} generations -> {a.out}")


if __name__ == "__main__":
    main()
