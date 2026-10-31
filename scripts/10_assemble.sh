#!/usr/bin/env bash
# Final assembly: contract-only metrics on the clean 256-gen dumps + one
# PPT-ready results file. Runs after chains 07/08/09.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/night4.log
stage() { echo "--- $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
echo "=== chain 4 queued $(date '+%F %T') ===" | tee -a "$LOG"

stage "waiting for chains 07/08/09"
while pgrep -f "07_overnight_dp[o]|08_eval_nigh[t]|09_ppt_clea[n]" > /dev/null; do sleep 60; done

stage "contract-only metrics on 256-gen dumps"
$PY -u -m duet.eval.prelim reports/dumps_256/*.jsonl \
    --out reports/final_prelim.md >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

stage "assemble RESULTS_FOR_PPT.md"
{
  echo "# DUET — results pack ($(date '+%F %H:%M'))"
  echo
  echo "All conditions: Qwen2.5-1.5B-Instruct + LoRA, same SFT init, same"
  echo "training set, same seed 42.  Metrics from the §5 dump contract only."
  echo
  echo "## Component A — isolation gate (checkpoints/critic, 20k rec x 1 ep)"
  echo
  echo '| metric | value | threshold |'
  echo '|---|---|---|'
  echo '| real-profile identification | 0.998 | >= 0.60 |'
  echo '| copy probe -> no-match | 1.000 | >= 0.80 |'
  echo '| generic reply -> no-match | 1.000 | >= 0.80 |'
  echo '| overall (chance 0.200) | 0.999 | — |'
  echo
  echo "## Table I metrics (harness, 256 generations/condition)"
  echo
  cat reports/harness_table.md 2>/dev/null || echo "(missing)"
  echo
  echo "## Contract-only metrics (same 256-gen dumps)"
  echo
  cat reports/final_prelim.md 2>/dev/null || echo "(missing)"
  echo
  echo "## DPO pair audits"
  echo
  grep -hE "mean (score|entailment) gap|pairs where rejected" \
      outputs/night2.log outputs/night3.log 2>/dev/null | tail -6
  echo
  echo "## Note on generation artifacts"
  echo
  echo "Dumps produced before the turn-boundary fix could roll into the next"
  echo "\`user:\` turn. Every number above uses trimmed generations for ALL"
  echo "conditions (same protocol), so comparisons are unaffected."
} > reports/RESULTS_FOR_PPT.md
echo "=== chain 4 finished $(date '+%F %T') ===" | tee -a "$LOG"
