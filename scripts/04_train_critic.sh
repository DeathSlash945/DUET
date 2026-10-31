#!/usr/bin/env bash
# Stage 04 — train the adversarial persona critic (Component A) + isolation validation.
#
# GATE before any critic-guided DPO (project_plan.md §6, Phase 1): the critic
# must beat chance by a wide margin on held-out dialogues AND route verbatim
# profile copies / generic replies to the no-match slot. train.py exits 2 on
# gate failure — do NOT proceed to pairs/DPO until it exits 0.
#
# Output: checkpoints/critic/<run>/final + LATEST symlink + validation_report.json
# Owner: Devanshu.
#
# Usage:
#   scripts/04_train_critic.sh                 # full run (configs/critic.yaml)
#   SMOKE=1 scripts/04_train_critic.sh         # 256 records, 1 epoch — pipeline check
#   DEVICE=cpu scripts/04_train_critic.sh      # no-GPU debugging
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # torch>=2.1: kills the frag-OOM seen in smoke
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv/bin/python"

test -x "$PY" || { echo "missing .venv — see README setup" >&2; exit 1; }
test -f data/processed/train.jsonl || { echo "missing data/processed/train.jsonl (stage 01)" >&2; exit 1; }

ARGS=(--config configs/critic.yaml --device "${DEVICE:-cuda}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  ARGS+=(--max-records 256 --epochs 1)
fi

echo "== critic training: ${ARGS[*]} =="
PYTHONPATH="$REPO/src" "$PY" -m duet.critic.train "${ARGS[@]}"
