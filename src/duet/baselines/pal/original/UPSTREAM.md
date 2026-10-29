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
2. `train_lora.py` + `decode.py` — same hardcoded lab prefix, removed.
3. `persona_datasets.py` (English class) — the `data_part` assert omits
   `"valid"`, but their own path-building branch handles it
   (`if data_part in ["train", "valid"]` → `valid_origin.txt`) and
   `trainers.py` hardcodes eval `split='valid'`. The assert is dead-wrong for
   English (upstream's own `valid_origin.txt` is unreachable without it);
   relaxed to include `"valid"`. The Chinese class has no such assert.
4. `trainers.py` (train + eval metric loops) — `batch_metrics[k].extend(v)`
   crashes on floats: `all_gather_if_needed` returns scalars unchanged at
   world_size=1, so `.tolist()` yields floats, not lists. Upstream ran 2 GPUs
   (their scripts set `CUDA_VISIBLE_DEVICES=1,2`), where the gather stacks
   scalars into lists. Upstream's own type-guard was sitting commented out
   right below each line — restored it.
5. `trainers.py` (FSDPTrainer.save) — `FullStateDictConfig(offload_to_cpu=True)`
   raises `NotImplementedError: offload_to_cpu=True and NO_SHARD is not
   supported yet` on torch 2.0.1 at world_size=1. Made world-size-aware
   (`offload_to_cpu=self.world_size > 1`) — multi-GPU behavior unchanged.

## Environment
Run with `.venv-pal` (upstream's pinned 2023 requirements) — NOT the main `.venv`.
All run adaptations (data paths, single GPU, wandb offline) go through hydra
CLI overrides in `scripts/`, never by editing files here (except patches listed above).

Upstream's requirements.txt omits two packages their own code needs, so we add:
- **peft==0.13.0** (imported by train_lora.py and trainers.py — the latter needs
  `peft.optimizers.create_loraplus_optimizer`, which only exists from 0.13.0;
  0.14.0+ imports `transformers.Cache` and breaks their transformers==4.29.2 pin)
- **accelerate==0.21.0** (era-correct; not listed upstream at all)
