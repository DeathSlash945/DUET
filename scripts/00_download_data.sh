#!/usr/bin/env bash
# Stage 00 — download PERSONA-CHAT (Original + Revised) and Dialogue NLI annotations.
# Owner: Kanishk (Phase 0). Output: data/raw/   Plan: project_plan.md §6
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

# PERSONA-CHAT both-format raw files -> data/raw/personachat/
# Official SHA256-pinned ParlAI archive first, Hugging Face mirror fallback
# (parl.ai answers 403 on some networks). Idempotent: skips files already present.
PY="$REPO/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "00_download_data: python interpreter not found in .venv" >&2; exit 1; }
PYTHONPATH="src" "$PY" - <<'PYEOF'
from duet.data.persona_chat import download_and_extract_parlai_data
path = download_and_extract_parlai_data("data/raw")
print(f"00_download_data: raw PERSONA-CHAT files ready under {path}")
PYEOF

# project_plan.md §9 also lists Dialogue NLI annotations (for baseline scorers
# and eval), but no download source is pinned anywhere in the repo yet.
# TODO(Dialogue NLI): add here once the team agrees on a source — do not guess a mirror.
echo "00_download_data: WARNING - Dialogue NLI annotations NOT downloaded (source not pinned yet; flagged to team)" >&2
