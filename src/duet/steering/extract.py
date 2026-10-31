"""
Component B: paired-activation collection for persona-attribute steering.

WHAT THIS COLLECTS
    For one persona attribute A and one transformer block L:

        direction[A][L] = mean(h_L | A expressed) - mean(h_L | A absent, matched)

    matching "normalization: mean_diff" in configs/steer.yaml. Directions are
    saved per layer to {activations_path}/{layer}/{attribute_id}.npz
    (project_plan.md §5.3 contract; {attribute} = the filename-safe
    attribute_id produced by duet.data.attributes).

PAIRED-CONTEXT CONSTRUCTION (documented per AGENTS.md)
    * Expressed context — a processed train record whose profile on
      `profile_side` (default 'self') contains attribute A. Presence is
      decided on canonical segment keys, identical to the inventory.
    * Matched context   — a record whose profiles on BOTH sides do NOT
      contain A. Candidates on each side are sorted by
      (len(history), len(profile sentences), id) and paired greedily 1:1 —
      deterministic approximate length matching, without replacement.
      Records that carry A on the other speaker's profile only are excluded
      from BOTH pools (contamination guard).
    * Prompt            — duet.training.prompting.format_prompt(...): the
      exact context the policy sees (single source of truth).
    * Activation        — residual-stream vector at the LAST PROMPT TOKEN
      (final non-padding position) of each block. Layer index = transformer
      block index, i.e. HF hidden_states[layer + 1] (hidden_states[0] is the
      embedding output). Gold target/response tokens are never encoded, so
      directions cannot leak the response.

NORMALIZATION / STEERING SCALE
    * Extraction stores the RAW mean difference (float32, pre-L2), per the
      config's "mean_diff". Unit-direction scaling by steering magnitude is a
      property of apply.py/sweep.py (later Phase 2) and is not decided here.

SMOKE MODE (CPU, no downloads)
    `python -m duet.steering.extract --smoke` builds a tiny random-weight
    GPT-2-shaped causal LM plus a WordLevel tokenizer trained locally on the
    prompts themselves, runs the full pipeline for a couple of inventory
    attributes, and writes npz files to outputs/smoke_attrs/. This validates
    PLUMBING ONLY — it says nothing about steering quality (AGENTS.md: do not
    claim steering works until measured on the real policy).

Owner: Kanishk (Component B). torch/transformers/tokenizers are imported
lazily so pairing, selection, and normalization stay testable in CPU-only
environments without the ML stack installed.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from duet.data.attributes import (
    AttributeInventory,
    AttributeStats,
    load_inventory,
    segment_profile,
)
from duet.data.schemas import ProcessedRecord, DatasetVersion
from duet.training.prompting import format_prompt

# Defaults for the configs/steer.yaml `extract:` section (config overrides them).
_EXTRACT_DEFAULTS: Dict[str, Any] = {
    "activations_path": "data/attrs",
    "pairs_per_attribute": 256,
    "normalization": "mean_diff",
    "profile_side": "self",
    "activation": "last_token",
    "context_length": 256,
    "selection_top_k": 0,
    "batch_size": 8,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_steer_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the full Component B YAML (configs/steer.yaml)."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Steering config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Steering config must be a mapping: {path}")
    return data


def extract_settings(steer_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge and validate the `extract:` section of the steering config."""
    section = steer_config.get("extract") or {}
    if not isinstance(section, dict):
        raise ValueError("steer_config['extract'] must be a mapping.")
    unknown = set(section) - set(_EXTRACT_DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown extract key(s): {sorted(unknown)}")

    cfg = {**_EXTRACT_DEFAULTS, **section}
    if cfg["normalization"] != "mean_diff":
        raise ValueError(
            f"Unsupported normalization {cfg['normalization']!r}; only 'mean_diff' is implemented."
        )
    if cfg["profile_side"] not in ("self", "partner", "either"):
        raise ValueError("profile_side must be 'self', 'partner' or 'either'.")
    if cfg["activation"] != "last_token":
        raise ValueError(
            f"Unsupported activation {cfg['activation']!r}; only 'last_token' is implemented."
        )
    for key in ("pairs_per_attribute", "context_length", "batch_size"):
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"extract.{key} must be an integer >= 1.")
    if not isinstance(cfg["selection_top_k"], int) or cfg["selection_top_k"] < 0:
        raise ValueError("extract.selection_top_k must be an integer >= 0.")
    if not isinstance(cfg["activations_path"], str) or not cfg["activations_path"]:
        raise ValueError("extract.activations_path must be a non-empty path string.")
    return cfg


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def profile_attribute_set(
    profile: List[str], *, split_compound: bool = False
) -> frozenset:
    """Canonical attribute keys of one profile (role does not affect keys)."""
    return frozenset(
        seg.canonical
        for seg in segment_profile(profile, role="self", split_compound=split_compound)
    )


@dataclass
class PairingResult:
    """Greedy length-matched expressed/matched pairs for one attribute.

    Attributes:
        pairs: (expressed, matched) record tuples, aligned 1:1.
        n_expressed: expressed candidates before the max_pairs cut.
        n_matched: matched candidates before the max_pairs cut.
    """
    pairs: List[Tuple[ProcessedRecord, ProcessedRecord]]
    n_expressed: int
    n_matched: int


def build_pairs(
    records: Iterable[ProcessedRecord],
    canonical: str,
    *,
    profile_side: str = "self",
    max_pairs: int = 256,
    split_compound: bool = False,
) -> PairingResult:
    """Build expressed vs matched context pairs for one attribute.

    See module docstring for the construction rules (contamination guard,
    deterministic length-sorted greedy matching, empty-history skip).
    """
    if profile_side not in ("self", "partner", "either"):
        raise ValueError("profile_side must be 'self', 'partner' or 'either'.")
    if not canonical or not isinstance(canonical, str):
        raise ValueError("canonical must be a non-empty string.")
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1:
        raise ValueError("max_pairs must be an integer >= 1.")

    cache: Dict[Tuple[str, ...], frozenset] = {}

    def _attrs(profile: List[str]) -> frozenset:
        key = tuple(profile)
        value = cache.get(key)
        if value is None:
            value = profile_attribute_set(profile, split_compound=split_compound)
            cache[key] = value
        return value

    expressed: List[ProcessedRecord] = []
    matched: List[ProcessedRecord] = []
    for record in records:
        if not record.history:
            continue  # nothing renderable as a dialogue context
        attrs_self = _attrs(record.profile_self)
        attrs_partner = _attrs(record.profile_partner)
        if profile_side == "self":
            on_side = canonical in attrs_self
        elif profile_side == "partner":
            on_side = canonical in attrs_partner
        else:
            on_side = canonical in attrs_self or canonical in attrs_partner

        if on_side:
            expressed.append(record)
        elif canonical not in attrs_self and canonical not in attrs_partner:
            matched.append(record)
        # else: attribute on the OTHER side only -> excluded from both pools.

    sort_key = lambda r: (len(r.history), len(r.profile_self), r.id)
    expressed.sort(key=sort_key)
    matched.sort(key=sort_key)
    n = min(len(expressed), len(matched), max_pairs)
    return PairingResult(
        pairs=list(zip(expressed[:n], matched[:n])),
        n_expressed=len(expressed),
        n_matched=len(matched),
    )


def select_attributes(
    inventory: AttributeInventory,
    *,
    profile_side: str = "self",
    pairs_per_attribute: int = 256,
    top_k: int = 0,
) -> List[AttributeStats]:
    """Inventory attributes with enough record-weighted contexts to extract.

    Sorted by the side-specific count (desc), then canonical — deterministic.
    """
    key = {"self": "n_self", "partner": "n_partner", "either": "count"}.get(profile_side)
    if key is None:
        raise ValueError("profile_side must be 'self', 'partner' or 'either'.")
    candidates = [
        stat for stat in inventory.attributes
        if getattr(stat, key) >= pairs_per_attribute
    ]
    candidates.sort(key=lambda stat: (-getattr(stat, key), stat.canonical))
    return candidates[:top_k] if top_k > 0 else candidates


# ---------------------------------------------------------------------------
# Prompts, activations, direction math
# ---------------------------------------------------------------------------

def render_prompt(record: ProcessedRecord) -> str:
    """The exact policy prompt for a record (shared training source of truth)."""
    return format_prompt(record.history, record.profile_self, record.profile_partner)


def model_layers(model) -> List[int]:
    """Transformer block indices of a HF causal LM (works for GPT-2 and Qwen)."""
    config = model.config
    n = getattr(config, "num_hidden_layers", None)
    if n is None:
        n = getattr(config, "n_layer", None)
    if n is None:
        raise ValueError("Cannot determine layer count from model.config.")
    return list(range(int(n)))


def collect_last_token_activations(
    model,
    tokenizer,
    prompts: List[str],
    *,
    layers: List[int],
    context_length: int,
    batch_size: int,
    device: str = "cpu",
) -> Dict[int, np.ndarray]:
    """Residual-stream vectors at the final prompt token, per requested block.

    Returns {layer: array[n_prompts, d_model] float32}.
    """
    import torch

    if not prompts:
        raise ValueError("collect_last_token_activations(): empty prompt list.")
    valid_layers = set(model_layers(model))
    bad = [int(L) for L in layers if int(L) not in valid_layers]
    if bad:
        raise ValueError(f"layer index out of range [0, {len(valid_layers)}): {bad}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    rows_by_layer: Dict[int, List[np.ndarray]] = {int(L): [] for L in layers}
    model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=context_length,
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            hidden = model(**enc, output_hidden_states=True).hidden_states
            # Last non-padding token per row (prompts are right-padded).
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            row_idx = torch.arange(enc["input_ids"].shape[0], device=device)
            for layer in rows_by_layer:
                # hidden_states[0] is the embedding output -> block L = index L+1.
                block_hidden = hidden[int(layer) + 1]
                vectors = block_hidden[row_idx, last_idx]
                rows_by_layer[int(layer)].append(vectors.float().cpu().numpy())
    return {
        layer: np.concatenate(chunks, axis=0)
        for layer, chunks in rows_by_layer.items()
    }


def mean_diff(
    expressed: Dict[int, np.ndarray],
    matched: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """mean(expressed) - mean(matched) per layer, float32 (config: mean_diff).

    With 1:1 paired selection this equals the mean of per-pair differences.
    """
    if not expressed or set(expressed) != set(matched):
        raise ValueError("expressed and matched must be non-empty with identical layers.")
    directions: Dict[int, np.ndarray] = {}
    for layer, exp in expressed.items():
        mat = matched[layer]
        if exp.shape != mat.shape:
            raise ValueError(
                f"layer {layer}: expressed shape {exp.shape} != matched shape {mat.shape}"
            )
        if exp.ndim != 2 or exp.shape[0] == 0:
            raise ValueError(f"layer {layer}: expected a non-empty 2-D array.")
        directions[int(layer)] = (exp.mean(axis=0) - mat.mean(axis=0)).astype(np.float32)
    return directions


# ---------------------------------------------------------------------------
# npz contract: {activations_path}/{layer}/{attribute_id}.npz
# ---------------------------------------------------------------------------

def save_direction(
    path: Union[str, Path],
    *,
    direction: np.ndarray,
    layer: int,
    attribute_id: str,
    canonical: str,
    n_pairs: int,
    normalization: str = "mean_diff",
    model_id: str = "",
    profile_side: str = "self",
    dataset_version: str = "",
    activation: str = "last_token",
) -> str:
    """Save one layer's direction plus provenance metadata as npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        direction=np.asarray(direction, dtype=np.float32),
        layer=np.asarray(int(layer), dtype=np.int64),
        attribute_id=np.asarray(str(attribute_id)),
        canonical=np.asarray(str(canonical)),
        n_pairs=np.asarray(int(n_pairs), dtype=np.int64),
        normalization=np.asarray(str(normalization)),
        model_id=np.asarray(str(model_id)),
        profile_side=np.asarray(str(profile_side)),
        dataset_version=np.asarray(str(dataset_version)),
        activation=np.asarray(str(activation)),
    )
    return str(path)


def load_direction(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load an npz written by save_direction() (no pickle)."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def extract_attribute(
    model,
    tokenizer,
    pairs: List[Tuple[ProcessedRecord, ProcessedRecord]],
    *,
    layers: List[int],
    attribute_id: str,
    canonical: str,
    out_dir: Union[str, Path],
    context_length: int,
    batch_size: int,
    device: str = "cpu",
    model_id: str = "",
    profile_side: str = "self",
    dataset_version: str = "",
) -> Dict[int, str]:
    """Full extraction for one attribute: pair -> activations -> mean_diff -> npz.

    Returns {layer: saved_path}.
    """
    if not pairs:
        raise ValueError(f"extract_attribute({canonical!r}): no pairs to extract from.")
    expressed_prompts = [render_prompt(exp) for exp, _ in pairs]
    matched_prompts = [render_prompt(mat) for _, mat in pairs]

    common = dict(
        layers=layers, context_length=context_length,
        batch_size=batch_size, device=device,
    )
    expressed_acts = collect_last_token_activations(
        model, tokenizer, expressed_prompts, **common
    )
    matched_acts = collect_last_token_activations(
        model, tokenizer, matched_prompts, **common
    )
    directions = mean_diff(expressed_acts, matched_acts)

    saved: Dict[int, str] = {}
    for layer, vector in directions.items():
        saved[int(layer)] = save_direction(
            Path(out_dir) / str(int(layer)) / f"{attribute_id}.npz",
            direction=vector,
            layer=int(layer),
            attribute_id=attribute_id,
            canonical=canonical,
            n_pairs=len(pairs),
            model_id=model_id,
            profile_side=profile_side,
            dataset_version=dataset_version,
        )
    return saved


# ---------------------------------------------------------------------------
# Smoke mode: tiny random model + local tokenizer (no downloads, CPU only)
# ---------------------------------------------------------------------------

def build_tiny_tokenizer(
    corpus: Iterable[str], *, vocab_size: int = 2000
):
    """WordLevel tokenizer trained locally on the prompts themselves."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[EOS]"]
    )
    tokenizer.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
    )


def build_tiny_model(
    tokenizer,
    *,
    n_layer: int = 2,
    n_embd: int = 32,
    n_head: int = 2,
    context_length: int = 64,
    seed: int = 0,
    device: str = "cpu",
):
    """Random-weight GPT-2-shaped causal LM — pipeline validation only."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=context_length,
        n_ctx=context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.to(device)
    model.eval()
    return model


def run_smoke(
    out_dir: Union[str, Path] = "outputs/smoke_attrs",
    *,
    seed: int = 0,
    pairs: int = 4,
    top_k: int = 2,
    version: DatasetVersion = "original",
    processed_dir: Union[str, Path] = "data/processed",
    inventory_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> List[str]:
    """End-to-end smoke run on real records with a tiny random model.

    Uses REAL inventory attributes and REAL processed records so the pairing
    and file contracts are exercised; only the model is synthetic. Writes to
    out_dir (never the real data/attrs activations path).
    """
    processed_dir = Path(processed_dir)
    records_path = processed_dir / version / "train.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing {records_path} — build processed data first (scripts/01_build_dataset.sh)."
        )
    with open(records_path, "r", encoding="utf-8") as f:
        records = [ProcessedRecord.from_dict(json.loads(line)) for line in f]

    inventory = load_inventory(
        inventory_path or Path("data/attrs") / f"inventory_{version}.json"
    )
    stats = select_attributes(
        inventory, profile_side="self", pairs_per_attribute=pairs, top_k=top_k
    )
    if not stats:
        raise RuntimeError(
            f"smoke: no attribute has >= {pairs} self-side contexts in the inventory."
        )

    planned: List[Tuple[AttributeStats, PairingResult]] = []
    for stat in stats:
        pairing = build_pairs(
            records, stat.canonical, profile_side="self", max_pairs=pairs
        )
        if not pairing.pairs:
            raise RuntimeError(f"smoke: pairing came up empty for {stat.canonical!r}.")
        planned.append((stat, pairing))

    corpus = [render_prompt(record) for record in records[:200]]
    for _, pairing in planned:
        corpus.extend(render_prompt(exp) for exp, _ in pairing.pairs)
        corpus.extend(render_prompt(mat) for _, mat in pairing.pairs)

    tokenizer = build_tiny_tokenizer(corpus)
    model = build_tiny_model(tokenizer, seed=seed, device=device)
    layers = model_layers(model)

    saved_paths: List[str] = []
    for stat, pairing in planned:
        saved = extract_attribute(
            model, tokenizer, pairing.pairs,
            layers=layers,
            attribute_id=stat.attribute_id,
            canonical=stat.canonical,
            out_dir=out_dir,
            context_length=64,
            batch_size=4,
            device=device,
            model_id=f"tiny-gpt2-random-seed{seed}",
            profile_side="self",
            dataset_version=version,
        )
        saved_paths.extend(saved.values())
        print(
            f"[extract:smoke] {stat.canonical}: pairs={len(pairing.pairs)} "
            f"(expressed={pairing.n_expressed}, matched={pairing.n_matched}) "
            f"layers={sorted(saved)} -> {out_dir}"
        )
    return sorted(saved_paths)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> List[ProcessedRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — build processed data first (scripts/01_build_dataset.sh)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [ProcessedRecord.from_dict(json.loads(line)) for line in f]


def _load_policy(model_ref: str, device: str):
    """Load a HF causal LM. CUDA extraction uses FP16 to fit 4 GB GPUs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: Dict[str, Any] = {}
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA extraction requested, but this PyTorch build cannot access CUDA. "
                "Install a CUDA-enabled PyTorch wheel before retrying."
            )
        # Keep policy weights in FP16.  The activations saved by this module
        # are still converted to float32 before the direction is written.
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["low_cpu_mem_usage"] = True
    print(f"[extract] loading model: {model_ref} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect paired activations and extract per-attribute steering directions."
    )
    parser.add_argument("--config", default="configs/steer.yaml")
    parser.add_argument("--version", choices=["original", "revised"], default="original")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny random local model (CPU, no downloads) — plumbing check only.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override extract.selection_top_k (0 = all eligible).")
    parser.add_argument("--attributes", nargs="+", default=None,
                        help="Exact canonical attribute strings to extract (bypasses selection).")
    parser.add_argument("--pairs", type=int, default=None,
                        help="Override extract.pairs_per_attribute.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Forward-pass batch size (default: 1 for 4 GB GPU pilots).")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layer/block indices (default: all model layers).")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (cuda needs approval).")
    parser.add_argument("--out", default=None,
                        help="Override extract.activations_path (smoke defaults to outputs/smoke_attrs).")
    parser.add_argument("--model", default=None,
                        help="Override model id (default: policy_ckpt if present, else base_model).")
    args = parser.parse_args()

    if args.smoke:
        out_dir = args.out or "outputs/smoke_attrs"
        paths = run_smoke(out_dir, version=args.version, device=args.device)
        print(f"[extract:smoke] wrote {len(paths)} direction file(s) to {out_dir} "
              "(random weights - plumbing only, NOT a steering result)")
        return

    steer_cfg = load_steer_config(args.config)
    cfg = extract_settings(steer_cfg)
    top_k = cfg["selection_top_k"] if args.top_k is None else args.top_k
    pairs_per_attribute = cfg["pairs_per_attribute"] if args.pairs is None else args.pairs
    if pairs_per_attribute < 1:
        raise SystemExit("--pairs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    out_dir = Path(args.out or cfg["activations_path"])

    model_ref = args.model
    if model_ref is None:
        policy_ckpt = steer_cfg.get("policy_ckpt")
        model_ref = policy_ckpt if policy_ckpt and Path(policy_ckpt).exists() else steer_cfg.get("base_model")
    if not model_ref:
        raise SystemExit("No model: set --model, policy_ckpt, or base_model in the config.")

    records = _load_records(Path("data/processed") / args.version / "train.jsonl")
    inventory = load_inventory(Path("data/attrs") / f"inventory_{args.version}.json")

    if args.attributes:
        known = {stat.canonical: stat for stat in inventory.attributes}
        unknown = [c for c in args.attributes if c not in known]
        if unknown:
            raise SystemExit(f"Unknown attribute canonical(s): {unknown}")
        stats = [known[c] for c in args.attributes]
    else:
        stats = select_attributes(
            inventory,
            profile_side=cfg["profile_side"],
            pairs_per_attribute=pairs_per_attribute,
            top_k=top_k,
        )
    if not stats:
        raise SystemExit(
            f"No inventory attribute has >= {pairs_per_attribute} contexts "
            f"on side '{cfg['profile_side']}' — lower pairs_per_attribute or check the inventory."
        )

    model, tokenizer = _load_policy(model_ref, args.device)
    layers = model_layers(model) if args.layers is None else args.layers

    n_saved = 0
    n_skipped = 0
    for stat in stats:
        pairing = build_pairs(
            records, stat.canonical,
            profile_side=cfg["profile_side"], max_pairs=pairs_per_attribute,
        )
        if not pairing.pairs:
            n_skipped += 1
            print(f"[extract] SKIP {stat.canonical}: "
                  f"expressed={pairing.n_expressed} matched={pairing.n_matched}")
            continue
        saved = extract_attribute(
            model, tokenizer, pairing.pairs,
            layers=layers,
            attribute_id=stat.attribute_id,
            canonical=stat.canonical,
            out_dir=out_dir,
            context_length=cfg["context_length"],
            batch_size=args.batch_size,
            device=args.device,
            model_id=model_ref,
            profile_side=cfg["profile_side"],
            dataset_version=args.version,
        )
        n_saved += len(saved)
        print(
            f"[extract] {stat.canonical}: pairs={len(pairing.pairs)} "
            f"layers={sorted(saved)} -> {out_dir}"
        )

    if n_saved == 0:
        raise SystemExit(f"[extract] FAILED: no direction files written (skipped {n_skipped}).")
    print(f"[extract] done: {n_saved} file(s) written, {n_skipped} attribute(s) skipped.")


if __name__ == "__main__":
    main()
