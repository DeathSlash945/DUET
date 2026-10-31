"""Entailment-signal preference pairs from candidate dumps (PAL-style).

Mirror of ``duet.critic.rewards`` so the two DPO arms differ in *signal only*:

    critic arm     : chosen = gold, rejected = argmin P_critic(true profile | y)
    entailment arm : chosen = gold, rejected = argmin P_NLI(entailment | profile, y)

Both read the same §5 generations dump (same candidate pool: K samples per
dialog) and write the same §5 DPO pair contract, so the downstream DPO run is
identical apart from the pairing signal.

    python -m duet.baselines.entailment_pairs \
        --generations data/dpo/cand_pool_k4.jsonl \
        --out data/dpo/dpo_entailment.jsonl
"""

import argparse

from ..eval.nli import DEFAULT_MODEL, load_nli, score_pairs, scrub
from ..training.common import read_jsonl, write_jsonl
from ..training.prompting import format_prompt


def build_pairs(records_by_id, generations, model, tok, device,
                batch_size=32) -> list:
    """Group sampled responses by dialog; pair gold vs least-entailed sample."""
    by_dialog: dict[str, list[str]] = {}
    for row in generations:
        by_dialog.setdefault(row["dialog_id"], []).append(row["response"])

    ids = sorted(by_dialog)
    pairs_in = []
    for dialog_id in ids:
        rec = records_by_id[dialog_id]
        premise = "\n".join(rec["profile_self"])
        for resp in by_dialog[dialog_id]:
            pairs_in.append((premise, resp))
        # gold is scored with the same premise for the audit gap
        pairs_in.append((premise, rec["target"]))

    scores = score_pairs(model, tok, pairs_in, device=device,
                         batch_size=batch_size)

    out = []
    pos = 0
    for dialog_id in ids:
        rec = records_by_id[dialog_id]
        responses = by_dialog[dialog_id]
        resp_scores = [scrub(scores[pos + i], "entailment")
                       for i in range(len(responses))]
        gold_score = scrub(scores[pos + len(responses)], "entailment")
        pos += len(responses) + 1

        worst = min(range(len(responses)), key=lambda i: resp_scores[i])
        best = max(range(len(responses)), key=lambda i: resp_scores[i])
        out.append({
            "prompt": format_prompt(rec["history"], rec["profile_self"],
                                    rec.get("profile_partner")),
            "chosen": rec["target"],
            "rejected": responses[worst],
            "signal": "entailment",
            "score_chosen": gold_score,
            "score_rejected": resp_scores[worst],
            "score_best_sample": resp_scores[best],
            "n_samples": len(responses),
            "dialog_id": dialog_id,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dpo_entailment.yaml")
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", default="data/dpo/dpo_entailment.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="NLI cross-encoder (default: nli-roberta-base)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    from ..training.common import cfg_get, load_yaml, to_ns
    cfg = to_ns(load_yaml(a.config))
    records = read_jsonl(cfg.data.train_path)
    records_by_id = {r["id"]: r for r in records}

    model, tok, _ = load_nli(a.model)
    model.to(a.device)
    pairs = build_pairs(records_by_id, read_jsonl(a.generations), model, tok,
                        a.device, batch_size=a.batch_size)
    write_jsonl(a.out, pairs)

    gaps = [p["score_chosen"] - p["score_rejected"] for p in pairs]
    print(f"wrote {len(pairs)} entailment pairs -> {a.out}")
    print(f"mean entailment gap (gold - rejected): "
          f"{sum(gaps) / len(gaps):.3f}")
    print(f"pairs where rejected out-scores gold: "
          f"{sum(1 for g in gaps if g < 0)}/{len(pairs)}")


if __name__ == "__main__":
    main()
