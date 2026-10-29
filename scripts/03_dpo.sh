#!/usr/bin/env bash
# Stage 03 — preference optimization (our stack: TRL DPO).
#   $1=configs/dpo_entailment.yaml  baseline control (PAL/PPT signal)
#   $1=configs/dpo_critic.yaml      Component A (identification signal)
#   $1=configs/bidir.yaml           Component C (both profiles)
# Matched conditions: same base model, SFT init, data, budget — only the signal differs.
# Pairs: data/dpo/<run_name>.jsonl — rows {prompt, chosen, rejected}, built by the
# signal's pair builder BEFORE this stage (PAL/PPT builders, critic/rewards.py).
# Usage: scripts/03_dpo.sh [config]   SMOKE=1 for a short run
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/dpo_entailment.yaml}"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--max-examples 64 --dump-generations 8)
fi

.venv/bin/python -m duet.training.run_dpo --config "$CONFIG" "${EXTRA[@]}"
