#!/usr/bin/env bash
# Stage 02c — PAL reproduction, Step 2: beam-search generations that become the
# DPO "rejected" candidates (upstream decode.py, data_mode=predict_select_then_generate).
#
# Input:  SFT run dir (SFT_RUN=... or newest outputs/pal_runs/pal-sft-gpt2-origin-seed42_*)
# Output: outputs/pal_runs/pal-gpt2-dpo-samples.json
#         ({prompt, golden_response, generated_response} joined by prompt string later)
#
# Deviation: their gpt2.sh omits data_split (decode.py then defaults to "test"),
# but the DPO stage trains on the VALID split and joins rejects by prompt string
# -> we pin data_split=valid so the join is exact.
# WANDB_MODE=offline: decode.py calls wandb.init unconditionally (no enabled check).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv-pal/bin/python"
UP="$REPO/src/duet/baselines/pal/original"
RUNS="$REPO/outputs/pal_runs"

SFT_RUN="${SFT_RUN:-$(ls -td "$RUNS"/pal-sft-gpt2-origin-seed42_* 2>/dev/null | head -1 || true)}"
test -n "$SFT_RUN" && test -f "$SFT_RUN/LATEST/policy.pt" || {
  echo "no SFT run with LATEST/policy.pt found — run scripts/02b_pal_sft.sh first (or set SFT_RUN=...)" >&2
  exit 1
}

echo "== PAL DPO samples: from $SFT_RUN =="
cd "$UP"
WANDB_MODE=offline CONFIG_PATH="$SFT_RUN" "$PY" decode.py \
  strategy=beam_search num_beam=3 max_new_tokens=40 \
  exp_name=pal-gpt2-dpo-samples \
  evaluate_model_path="$SFT_RUN/LATEST/policy.pt" \
  data_split=valid \
  data_mode=predict_select_then_generate \
  eval_batch_size=16 n_epochs=1 \
  wandb.project=persona_dialogue_decode \
  hydra.run.dir="$RUNS/hydra/pal-gpt2-dpo-samples"
echo "== done: $RUNS/pal-gpt2-dpo-samples.json =="
