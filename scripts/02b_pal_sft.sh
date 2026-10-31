#!/usr/bin/env bash
# Stage 02b — PAL reproduction, Step 1: multi-task SFT (upstream train.py).
#
# Faithful to upstream scripts/gpt2.sh "Step 1", with three documented
# adaptation classes (see src/duet/baselines/pal/original/UPSTREAM.md):
#   1. MODEL:  HF hub id "gpt2" (their local copy was named gpt2-small;
#              our train.py patch loads configured ids directly)
#   2. DATA:   English ORIGIN split — the files published with their repo
#              (train/valid_origin, valid#1, valid#2, test). Their gpt2.sh
#              default of persona_chat_revised needs train_revised.txt,
#              which is not distributed with the code.
#   3. RUNTIME: single RTX 4060 (their scripts assume 2 GPUs), wandb off,
#               all artifacts under outputs/ (gitignored).
#
# Usage:
#   scripts/02b_pal_sft.sh            # full run: their hyperparams (10 epochs)
#   SMOKE=1 scripts/02b_pal_sft.sh    # 512 examples — end-to-end pipeline check
#
# OOM knobs — their semantics: micro_batch = batch_size / (gradient_accumulation_steps * num_gpus),
#   and EFFECTIVE batch per step = batch_size (GPU-count independent). So to keep
#   their published effective batch of 128 on ONE gpu with a safe micro-batch of 4,
#   use batch_size=128 + gradient_accumulation_steps=32. If still OOM:
#   activation_checkpointing=true ... then halve max_length.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1   # stats must hit the log immediately (buffered stdout hid heartbeats)
REPO="$PWD"
PY="$REPO/.venv-pal/bin/python"
UP="$REPO/src/duet/baselines/pal/original"
DATA="$REPO/data/raw/pal_format"
RUNS="$REPO/outputs/pal_runs"
# (no PYTORCH_CUDA_ALLOC_CONF: torch 2.0.1 rejects the newer options)

test -x "$PY" || { echo "missing .venv-pal — install with:" >&2; echo "  python3.10 -m venv .venv-pal && .venv-pal/bin/pip install -r src/duet/baselines/pal/original/requirements.txt" >&2; exit 1; }
test -f "$DATA/train_origin.txt" || { echo "missing data: $DATA/train_origin.txt (see stage 00 / git history)" >&2; exit 1; }
mkdir -p "$RUNS"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EXP="pal-sft-gpt2-origin-SMOKE"
  # micro = 8/2 -> 4->2 (fp32 logits at seq 512 need micro<=2 on 8GB), eff batch 8.
  # eval_every=4096 skips mid-training evals (already proven at step-64 in a prior
  # run; their eval pass is 975 batches and dominates smoke wall-time).
  SCALE=(n_epochs=1 n_examples=128 batch_size=8 gradient_accumulation_steps=2 eval_batch_size=16 eval_every=4096 activation_checkpointing=true)
else
  EXP="pal-sft-gpt2-origin-seed42"
  # micro = 128/32 = 4 (OOM-safe with checkpointing), EFFECTIVE batch 128 = upstream's published value
  # eval_batch_size: upstream default 32 OOMs on 8GB — eval runs fp32
  # log_softmax over full logits (32 x seq x 50257 x 4B = 2.76 GiB) ON TOP of
  # the 4.2 GiB training state (killed the 2026-10-30 run at step 781).
  # At 4: ~0.35 GiB — safe. Eval also SAVES checkpoints, so it must survive.
  SCALE=(n_epochs=10 batch_size=128 gradient_accumulation_steps=32 eval_batch_size=4 activation_checkpointing=true)
fi

echo "== PAL SFT: exp=$EXP =="
cd "$UP"
"$PY" train.py \
  model=gpt2-small model.name_or_path=gpt2 \
  datasets=[persona_chat_origin] \
  loss=sft \
  exp_name="$EXP" \
  trainer=FSDPTrainer \
  data_mode=mixture \
  lr=2e-5 seed=42 \
  do_first_eval=false sample_during_eval=false \
  +is_chinese=false +is_llama3=false \
  +english_data_dir="$DATA" \
  local_dirs="[$RUNS]" \
  wandb.enabled=false \
  "${SCALE[@]}" \
  hydra.run.dir="$RUNS/hydra/${EXP}"
echo "== done: artifacts in $RUNS/${EXP}_* (look for LATEST/policy.pt) =="
