# Phase 0 — PAL reproduction: gate criteria

**Gate (project_plan.md §6):** baselines reproduce within a reasonable band.

## Target numbers — PAL paper, Table 2, GPT-2 rows

| System | BLEU-1 | BLEU-2 | ROUGE-L | Entropy | C.score |
|---|---|---|---|---|---|
| GPT-2 base | 5.97 | 2.50 | 9.74 | 5.18 | 0.333 |
| GPT-2 +FineTuning | 13.10 | 7.06 | 15.92 | 6.08 | 0.173 |
| GPT-2 +Prompting | 13.49 | 7.52 | 16.35 | 6.45 | 0.454 |
| **GPT-2 +PAL (their system)** | **17.05** | **8.77** | **16.66** | **6.93** | **0.811** |

(Revised split: +PAL = 15.94 / 7.89 / 14.55 / 6.75 / 0.427 — phase-4 robustness, not the Phase-0 gate.)

Intermediate sanity checkpoints toward the full system:
1. **SFT stage** (`02b_pal_sft.sh`): trains, evals, writes `<run>/LATEST/policy.pt` — ✅ smoke-verified; full run in progress.
2. **DPO stage** (`train_lora.py` + generated pairs): SFT→DPO hand-off loads `LATEST/policy.pt`.
3. **Decode stage** (`decode.py`): predictions on test/valid.
4. **Metrics**: BLEU/ROUGE/Entropy standard implementations; **C.score = NLI consistency** — their checkpoint is a lab-internal path (`/home/data_91_d/ligr/model/entailment_nli_model`); we substitute a public English NLI model, and this metric belongs to Sumagna's harness anyway (two seeded NLI checkpoints, selection/scoring separation).

## Documented deviations from their run

| Deviation | Why | Effect on comparability |
|---|---|---|
| English **origin** split (their `gpt2.sh` default is `persona_chat_revised`) | `train_revised.txt` is not distributed with their repo | Gate uses **Original** column above |
| Single GPU: `batch_size=128, gradient_accumulation_steps=32` (micro=4) vs their `4` (micro=32 @ 2 GPUs) | 8 GB VRAM | **Effective batch per update = 128 = theirs** (their semantics: eff = batch_size); only gradient-noise granularity differs |
| `activation_checkpointing=true` (their default false) | 8 GB VRAM | none on metrics |
| HF hub `gpt2` instead of their lab-local `gpt2-small` copy | lab filesystem doesn't exist | same architecture/weights |
| wandb disabled, paths via hydra overrides | single-user machine | none |

## Known upstream quirks (not fixed unless they block the gate)

- **`ppl/eval/chosen` can be `nan`**: `_get_batch_ppl` divides per row by non-masked label count; any fully-masked eval row (their `max_prompt_length=768` > `max_length=512` makes this possible) → 0/0 → nan poisons the batch mean. `loss/eval` is computed differently and stays valid — use loss, not their ppl.
- `eval` pass materializes **975 batches** (mixture mode loads select+generate datasets separately → ~2× the valid split).

## Reproduction checklist

- [ ] SFT full run (10 epochs) completes → `LATEST/policy.pt`
- [ ] `decode.py` generates DPO samples from SFT checkpoint
- [ ] `train_lora.py` DPO run (3 epochs, eff batch 128) → LoRA `LATEST`
- [ ] `decode.py` final predictions
- [ ] Metrics computed → compare against Table 2 Original column
- [ ] PPT baseline reimplementation (Takayama et al., COLING 2025 — ACL Anthology, fetch paper first)
