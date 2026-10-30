#!/usr/bin/env bash
# Stage 01 — build processed dataset: keeps BOTH profiles (schemas.py contract),
# segments attributes (Component B), builds critic candidate pools (Component A).
# Owner: Kanishk (Phase 0). Output: data/processed/, data/critic/, data/attrs/
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "01_build_dataset: python interpreter not found in .venv" >&2; exit 1; }

# Stage 1 — BOTH-profiles processed records (schemas.py §5 contract),
# Original + Revised -> data/processed/{original,revised}/{split}.jsonl
PYTHONPATH="src" "$PY" -m duet.data.persona_chat --version both \
  --raw-dir data/raw --output-dir data/processed

# Stage 2 — profile attribute segmentation (Component B input) ->
# data/attrs/inventory_{original,revised}.json  (params: configs/attributes.yaml)
PYTHONPATH="src" "$PY" -m duet.data.attributes --config configs/attributes.yaml

# PENDING — Stage 3: distractor pools + copy probes -> data/critic/ (Component A input;
#           final sampling strategy stays with Devanshu per AGENTS.md).
echo "01_build_dataset: stages 1-2 complete (processed records + attribute inventories); PENDING: data/critic/ (Component A pools)" >&2
