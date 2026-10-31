#!/usr/bin/env bash
# Clean PPT arm (turn-trimmed negatives) + final harness pass.
# Waits for 07 (untrimmed PPT arm) and 08 (K=4 arms + harness) to finish.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
LOG=outputs/night3.log
echo "=== chain 3 queued $(date '+%F %T') ===" | tee -a "$LOG"
stage() { echo "--- $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" >> "$LOG" 2>&1; echo "    exit=$?" | tee -a "$LOG"; }

stage "waiting for chains 07 and 08"
while pgrep -f "07_overnight_dp[o]|08_eval_nigh[t]" > /dev/null; do sleep 60; done

stage "clean PPT pairs (turn-trimmed negatives)"
run $PY -u -m duet.baselines.ppt.pairs --config configs/ppt.yaml \
    --model outputs/sft/final --limit 2500 --out data/dpo/ppt_pairs.jsonl

stage "PPT DPO arm"
run $PY -u -m duet.training.run_dpo --config configs/ppt.yaml \
    --pairs data/dpo/ppt_pairs.jsonl --dump-generations 64

stage "256-gen dumps: ppt + any missing conditions"
for c in ppt dpo_critic dpo_entailment; do
  [ -d "outputs/$c/final" ] || { echo "    skip $c" | tee -a "$LOG"; continue; }
  run $PY -u -m duet.eval.sample --adapter "outputs/$c/final" --limit 256 \
      --device cuda --condition "$c" --out "reports/dumps_256/$c.jsonl"
done

stage "final harness pass over all conditions"
run $PY -u -m duet.eval.harness --dumps reports/dumps_256/*.jsonl \
    --out reports/harness_table.md

stage "final summary"
{
  echo "# Final results (all arms) — $(date '+%F %H:%M')"
  echo
  cat reports/harness_table.md 2>/dev/null || echo "(missing)"
  echo
  echo "## Pair audits"
  grep -hE "mean (score|entailment) gap|pairs where rejected" outputs/night2.log outputs/night3.log 2>/dev/null | tail -6
} > reports/final_summary.md
echo "=== chain 3 finished $(date '+%F %T') ===" | tee -a "$LOG"
