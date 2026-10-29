#!/usr/bin/env bash
# Stage 02e — PAL reproduction, Step 4: final predictions + automatic metrics
# (decode.py computes BLEU-4, ROUGE-L, entropy itself and writes them into the
# output JSON — see decode.py cal_BLEU_4 / cal_entropy).
#
# LoRA caveat: the DPO run's LATEST/policy.pt is an EMPTY placeholder (upstream
# saves only the adapter) and decode.py has no PeftModel support — so we merge
# base+adapter first (duet.baselines.pal.merge_lora_ckpt, run with the MAIN venv)
# into LATEST/policy_merged.pt and point decode at that.
#
# C.score (NLI consistency) is NOT computed here — their NLI checkpoint is a lab
# path; it belongs to Aparajith's eval harness with public NLI models.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv-pal/bin/python"
UP="$REPO/src/duet/baselines/pal/original"
RUNS="$REPO/outputs/pal_runs"
SAMPLES="$RUNS/pal-gpt2-dpo-samples.json"

DPO_RUN="${DPO_RUN:-$(ls -td "$RUNS"/pal-gpt2-dpo-origin-seed42_* 2>/dev/null | head -1 || true)}"
test -n "$DPO_RUN" && test -f "$DPO_RUN/LATEST/adapter_config.json" || {
  echo "no DPO run with LATEST/adapter_config.json found — run scripts/02d_pal_dpo.sh first (or set DPO_RUN=...)" >&2; exit 1; }
test -f "$SAMPLES" || { echo "missing $SAMPLES — run scripts/02c_pal_dpo_samples.sh first" >&2; exit 1; }

echo "== merging LoRA adapter (main venv) =="
.venv/bin/python -m duet.baselines.pal.merge_lora_ckpt --run-dir "$DPO_RUN"

EXP="pal-gpt2-dpo-origin-predict"
echo "== PAL predict: exp=$EXP =="
cd "$UP"
WANDB_MODE=offline CONFIG_PATH="$DPO_RUN" "$PY" decode.py \
  strategy=beam_search num_beam=3 max_new_tokens=40 \
  exp_name="$EXP" \
  evaluate_model_path="$DPO_RUN/LATEST/policy_merged.pt" \
  data_mode=predict_select_then_generate \
  +select_data_path="$SAMPLES" \
  eval_batch_size=16 n_epochs=1 \
  wandb.project=persona_dialogue_decode \
  hydra.run.dir="$RUNS/hydra/${EXP}"
echo "== done: $RUNS/${EXP}.json (per-item bleu/rouge_L + corpus metrics printed above) =="
