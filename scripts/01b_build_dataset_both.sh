#!/usr/bin/env bash
# Stage 01b — build the BOTH-profile processed data (Component C, Devanshu).
#
# Source: original ParlAI PERSONA-CHAT *_both_original.txt (data/raw/
# personachat_original/) — the only mirrored format with BOTH speakers'
# personas per episode. Download (parl.ai is 403; HF mirror):
#   BASE=https://huggingface.co/datasets/awsaf49/persona-chat/resolve/main
#   for f in train valid test; do curl -L -o data/raw/personachat_original/${f}_both_original.txt $BASE/${f}_both_original.txt; done
#
# Output: data/processed/{train,valid,test}.jsonl — §5 contract, now with real
# profile_partner on every record + synthesized partner-perspective records +
# persona ids for profile-level holdout. Self records keep legacy ids, so the
# new files are a superset of the old PAL-format-derived ones.
# Owner: Devanshu (Component C). Kanishk's data pipeline supersedes when ready.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
test -f data/raw/personachat_original/train_both_original.txt || {
  echo "missing data/raw/personachat_original/ (see header download cmd)" >&2; exit 1; }
PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m duet.bidir.dataset \
  --src data/raw/personachat_original --out data/processed "$@"
