#!/usr/bin/env bash
# Stage 05 — extract per-attribute steering directions (paired activations,
# mean difference per layer). Output: data/attrs/{layer}/{attribute}.npz
# Owner: Kanishk (Phase 2). GATE: sweeps only after extraction validates.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || { echo "05_extract_directions: Python interpreter not found in .venv" >&2; exit 1; }

# Forward arguments verbatim.  Use --smoke for the CPU-only, no-download
# plumbing check; normal extraction may load a model and is approval-gated.
PYTHONPATH="src" "$PY" -m duet.steering.extract "$@"
