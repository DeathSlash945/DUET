#!/usr/bin/env bash
# Stage 90 — run the full evaluation harness over one or more generation dumps.
#   python -m duet.eval.harness --generations outputs/{run_id}/generations.jsonl
# Runs ALL metrics from project_plan.md §8 from generations files alone
# (identifiability, NLI×2, copy rate, sensitivity, partner suite, ledger, drift).
# Owner: Sumagna — consumes dumps only, never training code.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "90_run_eval: not yet implemented — owner Sumagna" >&2
exit 1
