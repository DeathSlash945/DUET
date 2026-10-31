#!/usr/bin/env bash
# Night 2 — entailment arm (NLI selector) + full evaluation harness.
#
# Waits for the first overnight chain (07) to finish, then:
#   A. K=4 candidate pool from pi_sft (bidir SFT)          [same pool for BOTH arms]
#   B. critic pairs   (rewards.py)        -> dpo_critic
#   C. entailment pairs (NLI cross-encoder) -> dpo_entailment
#   D. DPO critic arm      (beta .1, 3 ep, eff 16, seed 42)
#   E. DPO entailment arm  (identical recipe; signal-only contrast)
#   F. 256-generation dumps for every condition (fixed dump contract)
#   G. harness: Table I metrics (identifiability, copy, hallucination,
#      contradiction, partner recall, drift)
#
# Outputs: reports/harness_table.md (+ .tex rows), reports/night_summary.md
#
# Usage: setsid nohup systemd-inhibit --what=idle:sleep --mode=block \
#          --why="entailment arm + harness" bash scripts/08_eval_night.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
LOG=outputs/night2.log
N=${N:-2500}
mkdir -p outputs data/dpo reports/dumps_256
echo "=== night-2 run queued $(date '+%F %T') (N=$N) ===" | tee -a "$LOG"

stage() { echo "--- $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
run() { "$@" >> "$LOG" 2>&1; echo "    exit=$?" | tee -a "$LOG"; }

stage "waiting for the first overnight chain (07) to finish"
while pgrep -f "07_overnight_dp[o]" > /dev/null; do sleep 60; done
stage "first chain done; starting night 2"

stage "A/7 K=4 candidate pool (shared by both arms)"
run $PY -u -m duet.eval.sample --adapter outputs/sft/final --limit "$N" \
    --samples-per-dialog 4 --temperature 1.0 --device cuda \
    --condition sft_bidir --out data/dpo/cand_pool_k4.jsonl

stage "B/7 critic pairs from the K=4 pool"
run $PY -u -m duet.critic.rewards --config configs/critic.yaml \
    --generations data/dpo/cand_pool_k4.jsonl --out data/dpo/dpo_critic.jsonl

stage "C/7 entailment pairs (NLI cross-encoder) from the SAME pool"
run $PY -u -m duet.baselines.entailment_pairs \
    --generations data/dpo/cand_pool_k4.jsonl --out data/dpo/dpo_entailment.jsonl

stage "D/7 DPO critic arm"
run $PY -u -m duet.training.run_dpo --config configs/dpo_critic.yaml \
    --pairs data/dpo/dpo_critic.jsonl --dump-generations 64

stage "E/7 DPO entailment arm (identical recipe)"
run $PY -u -m duet.training.run_dpo --config configs/dpo_entailment.yaml \
    --pairs data/dpo/dpo_entailment.jsonl --dump-generations 64

stage "F/7 256-generation dumps for every condition"
for cond in sft:sft sft_selfonly:sft_selfonly dpo_critic:dpo_critic \
            dpo_entailment:dpo_entailment ppt:ppt; do
  run_dir=${cond%%:*}; cond_name=${cond##*:}
  [ -d "outputs/$run_dir/final" ] || { echo "    skip $run_dir (no checkpoint)" | tee -a "$LOG"; continue; }
  run $PY -u -m duet.eval.sample --adapter "outputs/$run_dir/final" \
      --limit 256 --device cuda --condition "$cond_name" \
      --out "reports/dumps_256/$cond_name.jsonl"
done

stage "G/7 harness over all condition dumps"
run $PY -u -m duet.eval.harness --dumps reports/dumps_256/*.jsonl \
    --out reports/harness_table.md

stage "summary"
{
  echo "# Night-2 results — $(date '+%F %H:%M')"
  echo
  echo "Arms: same policy, same SFT init, same K=4 candidate pool, same seed;"
  echo "critic vs NLI-entailment differ in the pairing signal only."
  echo
  echo "## Table I metrics (harness, 256 generations per condition)"
  echo
  cat reports/harness_table.md 2>/dev/null || echo "(missing)"
  echo
  echo "## DPO pair audits"
  echo
  grep -E "mean (score|entailment) gap|pairs where rejected" outputs/night2.log | tail -4
  echo
  echo "## Prelim contract-only (64-gen training dumps)"
  echo
  for f in reports/overnight_critic_arm.md reports/overnight_ppt_arm.md; do
    [ -f "$f" ] && { echo "### $f"; echo; cat "$f"; echo; }
  done
} > reports/night_summary.md

echo "=== night-2 finished $(date '+%F %T') ===" | tee -a "$LOG"
echo "See reports/night_summary.md and reports/harness_table.md" | tee -a "$LOG"
