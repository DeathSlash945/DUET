#!/usr/bin/env bash
# Overnight DPO arms (post-deadline fill for Table I's DPO rows).
#
# Produces, from the *same* pi_sft (bidir SFT) and the *same* candidate dumps:
#   1. critic arm   — pairs from the identification critic (rewards.py)
#   2. PPT arm      — pseudo-preference pairs (persona swap)
# Each arm: build pairs -> DPO (3 epochs, paper/config hyperparams) -> dump 64
# generations -> contract-only metrics (eval/prelim.py).
#
# NOT included (documented in reports/overnight_summary.md):
#   - entailment arm: needs a PAL checkpoint; PAL SFT was paused at 72,832
#     examples before its first eval/save boundary, so no sample exists yet.
#   - full harness table: owned by B24DS507, not merged.
#
# Usage:  setsid nohup systemd-inhibit --what=idle:sleep --mode=block \
#           --why="overnight DPO arms" bash scripts/07_overnight_dpo.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
LOG=outputs/overnight.log
N=${N:-2500}                       # dialogs per arm
mkdir -p outputs data/dpo reports
echo "=== overnight run started $(date '+%F %T') (N=$N) ===" | tee -a "$LOG"

stage() { echo "--- $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- critic arm
stage "1/5 sample $N candidates from pi_sft (bidir SFT)"
$PY -u -m duet.eval.sample --adapter outputs/sft/final --limit "$N" \
    --out data/dpo/cand_sft.jsonl --condition sft_bidir --device cuda >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

stage "2/5 critic-score candidates -> pairs"
$PY -u -m duet.critic.rewards --config configs/critic.yaml \
    --generations data/dpo/cand_sft.jsonl --out data/dpo/dpo_critic.jsonl \
    >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

stage "3/5 DPO critic arm (beta 0.1, 3 epochs)"
$PY -u -m duet.training.run_dpo --config configs/dpo_critic.yaml \
    --pairs data/dpo/dpo_critic.jsonl --dump-generations 64 >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

$PY -u -m duet.eval.prelim outputs/dpo_critic/generations.jsonl \
    --out reports/overnight_critic_arm.md > /dev/null 2>&1
echo "    prelim: reports/overnight_critic_arm.md (exit=$?)" | tee -a "$LOG"

# ------------------------------------------------------------------- PPT arm
stage "4/5 PPT pairs (persona-swap negatives, beta 0.5 / lr 1e-6 / 3 ep)"
$PY -u -m duet.baselines.ppt.pairs --config configs/ppt.yaml \
    --model outputs/sft/final --limit "$N" --out data/dpo/ppt_pairs.jsonl \
    >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

$PY -u -m duet.training.run_dpo --config configs/ppt.yaml \
    --pairs data/dpo/ppt_pairs.jsonl --dump-generations 64 >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

$PY -u -m duet.eval.prelim outputs/ppt/generations.jsonl \
    --out reports/overnight_ppt_arm.md > /dev/null 2>&1
echo "    prelim: reports/overnight_ppt_arm.md (exit=$?)" | tee -a "$LOG"

# ------------------------------------------------------------------- summary
stage "5/5 summary"
{
  echo "# Overnight DPO arms — $(date '+%F %H:%M')"
  echo
  echo "Same policy (Qwen2.5-1.5B-Instruct + LoRA), same SFT init (outputs/sft,",
  echo "bidir), same candidate dialogs (first $N of the train split), same seed."
  echo "Only the pair-selection signal differs."
  echo
  echo "## DPO critic arm (beta 0.1)"
  echo
  cat reports/overnight_critic_arm.md 2>/dev/null || echo "(missing)"
  echo
  echo "## PPT arm (beta 0.5, paper Appendix B)"
  echo
  cat reports/overnight_ppt_arm.md 2>/dev/null || echo "(missing)"
  echo
  echo "## Not run"
  echo
  echo "- entailment arm: blocked on a PAL checkpoint (PAL SFT paused at 72,832"
  echo "  examples, no eval/save boundary reached)."
  echo "- full harness columns (identifiability, copy/hallucination/contradiction,"
  echo "  partner recall, drift): owned by B24DS507, not merged."
  echo
  echo "## Reference: SFT conditions (same protocol)"
  echo
  echo '| condition | n | copy@1 | copy@4 | self_cov | partner_cov | resp_len |'
  echo '|---|---|---|---|---|---|---|'
  echo '| sft (bidir) | 64 | 0.198 | 0.005 | 0.123 | 0.086 | 47.500 |'
  echo '| sft_selfonly | 64 | 0.179 | 0.005 | 0.109 | -- | 43.703 |'
} > reports/overnight_summary.md

echo "=== overnight run finished $(date '+%F %T') ===" | tee -a "$LOG"
echo "See reports/overnight_summary.md" | tee -a "$LOG"
