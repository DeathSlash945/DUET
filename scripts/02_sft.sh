#!/usr/bin/env bash
# Stage 02 — supervised fine-tuning of the DUET policy (our stack: TRL + PEFT).
# Config: configs/sft.yaml   Owner: Devanshu
# Data:   data/processed/*.jsonl (§5 contract) — built by scripts/01 (Zala);
#         Phase-0 stand-in: python -m duet.baselines.pal.to_contract
# Usage:  scripts/02_sft.sh  |  SMOKE=1 scripts/02_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=(--dump-generations 64)
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--max-examples 96 --max-eval-examples 64 --dump-generations 8)
fi

.venv/bin/python -m duet.training.run_sft --config configs/sft.yaml "${EXTRA[@]}"
