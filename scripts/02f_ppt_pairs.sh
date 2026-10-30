#!/usr/bin/env bash
# Stage 02f — PPT baseline: pseudo-negative generation + pair building
# (Takayama et al., COLING 2025). NOT part of the PAL 02b-02e chain — runs
# once per DPO-ready SFT checkpoint, feeding stage 03 for the ppt arm only.
# GPU step: needs outputs/sft/LATEST (pi_sft); writes data/dpo/ppt.jsonl.
# Usage: scripts/02f_ppt_pairs.sh [extra args]  |  SMOKE=1 scripts/02f_ppt_pairs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--limit 64)
fi

.venv/bin/python -m duet.baselines.ppt.pairs --config configs/ppt.yaml "${EXTRA[@]}" "$@"
