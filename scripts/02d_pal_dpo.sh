#!/usr/bin/env bash
# Stage 02d — PAL reproduction, Step 3: persona alignment via LoRA DPO
# (upstream train_lora.py). Rejected candidates come from stage 02c; chosen =
# gold responses; joins by prompt string.
#
# Upstream Step-1 output is a FULL model, while train_lora.py's model.archive
# assumed an adapter dir — DUET PATCH #6 makes it load full weights + fresh
# trainable adapter (r=16, alpha=32, gaussian init — upstream's own else-branch
# hyperparameters).
#
# Hardware: eff batch 128 = upstream; micro = 128/32 = 4 with activation
# checkpointing (8 GB). OOM fallback: gradient_accumulation_steps=64 (micro=2).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv-pal/bin/python"
UP="$REPO/src/duet/baselines/pal/original"
DATA="$REPO/data/raw/pal_format"
RUNS="$REPO/outputs/pal_runs"
SAMPLES="$RUNS/pal-gpt2-dpo-samples.json"

SFT_RUN="${SFT_RUN:-$(ls -td "$RUNS"/pal-sft-gpt2-origin-seed42_* 2>/dev/null | head -1 || true)}"
test -n "$SFT_RUN" && test -f "$SFT_RUN/LATEST/policy.pt" || {
  echo "no SFT run found — run scripts/02b_pal_sft.sh first (or set SFT_RUN=...)" >&2; exit 1; }
test -f "$SAMPLES" || {
  echo "missing $SAMPLES — run scripts/02c_pal_dpo_samples.sh first" >&2; exit 1; }

EXP="pal-gpt2-dpo-origin-seed42"
echo "== PAL DPO: exp=$EXP, archive=$SFT_RUN/LATEST, rejects=$SAMPLES =="
cd "$UP"
"$PY" train_lora.py \
  model=gpt2-small model.name_or_path=gpt2 \
  datasets=[persona_chat_origin] \
  loss=dpo loss.beta=0.1 \
  exp_name="$EXP" \
  trainer=FSDPTrainer \
  batch_size=128 gradient_accumulation_steps=32 eval_batch_size=32 \
  activation_checkpointing=true \
  n_epochs=3 lr=1e-6 eval_every=5000 seed=42 \
  model.archive="$SFT_RUN/LATEST" \
  data_mode=golden_select_then_generate \
  dpo_data_mode=generate_for_dpo \
  dpo_path="$SAMPLES" \
  do_first_eval=false sample_during_eval=false \
  +is_chinese=false +is_llama3=false +is_lora=true \
  +english_data_dir="$DATA" \
  local_dirs="[$RUNS]" \
  wandb.enabled=false \
  hydra.run.dir="$RUNS/hydra/${EXP}"
echo "== done: $RUNS/${EXP}_*/LATEST/ (adapter + empty upstream policy.pt) =="
