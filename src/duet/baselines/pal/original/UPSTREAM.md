# Vendored upstream: kylokano/PAL

Source: https://github.com/kylokano/PAL (commit `cc8b8b5`)
Vendored: 2026-10-29

## Policy
Files here are kept as close to upstream as possible. Any deviation is listed
below and must be re-documented here.

## Patches applied (vs upstream)
1. `train.py` model-path handling — upstream hardcodes the prefix
   `/home/data_91_d/ligr/model/` in BOTH branches of the absolute-path check,
   so no model can ever load outside their lab filesystem. Patched to accept
   absolute paths and Hugging Face hub ids.

## Environment
Run with `.venv-pal` (upstream's pinned 2023 requirements) — NOT the main `.venv`.
All run adaptations (data paths, single GPU, wandb offline) go through hydra
CLI overrides in `scripts/`, never by editing files here (except patches listed above).
