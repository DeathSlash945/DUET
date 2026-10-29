#!/usr/bin/env bash
# Stage 03 — preference optimization.
#   --config configs/dpo_entailment.yaml  → baseline control (PAL/PPT signal)
#   --config configs/dpo_critic.yaml      → Component A (identification signal)
#   --config configs/bidir.yaml           → Component C (both profiles)
# Matched conditions: same base model, SFT init, data, budget — only signal differs.
# Owner: Devanshu. Output: outputs/{run_name}/
set -euo pipefail
cd "$(dirname "$0")/.."
echo "03_dpo: not yet implemented — Phase 0/1, owner Devanshu" >&2
exit 1
