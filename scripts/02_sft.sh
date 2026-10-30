#!/usr/bin/env bash
# Stage 02 — supervised fine-tuning of the DUET policy (our stack: TRL + PEFT).
# Config: configs/sft.yaml (default) or configs/sft_selfonly.yaml (Component C
#         baseline — same everything, profile_partner stripped)   Owner: Devanshu
# Data:   data/processed/*.jsonl (§5 contract) — stage 01b (both profiles)
# Usage:  scripts/02_sft.sh [config]  |  SMOKE=1 scripts/02_sft.sh [config]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/sft.yaml}"
EXTRA=(--dump-generations 64)
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--max-examples 96 --max-eval-examples 64 --dump-generations 8)
fi

.venv/bin/python -m duet.training.run_sft --config "$CONFIG" "${EXTRA[@]}"
