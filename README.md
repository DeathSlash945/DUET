# DUET

**Dual-persona Encoding and Traits for Persona-Aligned Dialogue Generation**
Team Phronesis — Devanshu Sabharwal, Kanishk Nandeshwar, S Aparajith · ML Course Project

Persona-aligned dialogue generation on PERSONA-CHAT. DUET tests three claims:

1. **The supervision signal is wrong.** An identification-based critic beats the standard entailment (NLI) reward under matched preference optimization.
2. **The representation is too coarse.** Persona attributes are separable directions in activation space — and they compose.
3. **The scope is too narrow.** A dialogue model trained with *both* speakers' profiles uses the partner's persona.

Base/reproduction target: [kylokano/PAL](https://github.com/kylokano/PAL). Full plan, ownership, phases, and decision gates: **[project_plan.md](project_plan.md)**.

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| `configs/` | shared | One YAML per experiment — every run is config-driven |
| `src/duet/data/` | Kanishk | PERSONA-CHAT pipeline (**keeps both profiles**), attribute segmentation, `schemas.py` contracts |
| `src/duet/baselines/` | Devanshu | PAL reproduction + PPT reimplementation |
| `src/duet/critic/` | Devanshu | Component A — adversarial persona critic |
| `src/duet/steering/` | Kanishk | Component B — attribute direction extraction + steering hooks |
| `src/duet/bidir/` | Devanshu | Component C — bidirectional persona training |
| `src/duet/training/` | shared | SFT / DPO wrappers over TRL + PEFT |
| `src/duet/eval/` | Aparajith | Evaluation harness (runs on generation dumps alone) |
| `scripts/` | shared | Runnable entry points, one per stage |
| `tests/` | shared | Schema tests + smoke tests |
| `reports/` | shared | Results tables, figures, weekly notes |

## Quickstart

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q
```

Raw data, outputs, and checkpoints are gitignored (`data/`, `outputs/`). Regenerate data from scripts — never commit it.

## Working agreements

- Feature branch per task (`devanshu/critic-train`, `zala/data-pipeline`, `sumagna/eval-harness`); one teammate review before merging to `main`.
- Every run reproducible from `configs/*.yaml` + a script.
- All models dump generations to `outputs/{run_id}/generations.jsonl` in the shared format (see `project_plan.md` §5) — the eval harness consumes that file and nothing else.

## Status

**Phase 0 — in progress** (data pipeline, PAL/PPT baselines, eval harness foundation).
