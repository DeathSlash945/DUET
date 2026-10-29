"""Shared plumbing for DUET training runs: config, data I/O, model/LoRA, generation dumps."""

import dataclasses
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model


# --------------------------------------------------------------------------- config

def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_ns(obj):
    """Recursively convert dicts to SimpleNamespace for cfg.train.lr style access."""
    from types import SimpleNamespace
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_ns(v) for v in obj]
    return obj


# readable config keys -> HF/TRL dataclass field names
FIELD_MAP = {
    "epochs": "num_train_epochs",
    "lr": "learning_rate",
    "per_device_batch_size": "per_device_train_batch_size",
    "grad_accum": "gradient_accumulation_steps",
}


def config_args(section: dict, cls) -> dict:
    """Map + intersect a config section with a dataclass's fields (keeps runs
    stable when configs grow keys the installed library version doesn't know)."""
    names = {f.name for f in dataclasses.fields(cls)}
    out = {}
    for k, v in section.items():
        k = FIELD_MAP.get(k, k)
        if k in names:
            out[k] = v
    return out


# --------------------------------------------------------------------------- data

def read_jsonl(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_init(path):
    """Resolve a checkpoint reference like outputs/sft/LATEST to a real dir."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"init_from does not exist: {path}")
    return str(p)


def cfg_get(obj, key, default=None):
    """Read key from a dict or SimpleNamespace interchangeably."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


# --------------------------------------------------------------------------- model

def load_policy(cfg, trainable: bool = True):
    """Load base model + tokenizer, attach (or continue) a LoRA adapter.

    cfg fields used: base_model, init_from (optional), lora.* (only when
    starting a fresh adapter), train.bf16, train.gradient_checkpointing.
    """
    dtype = torch.bfloat16 if cfg_get(cfg.train, "bf16", True) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    init = resolve_init(getattr(cfg, "init_from", None))
    if init:
        # Continue training an existing adapter (e.g. SFT -> DPO hand-off).
        model = PeftModel.from_pretrained(base, init, is_trainable=trainable)
    else:
        lora = cfg.lora
        model = get_peft_model(base, LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            task_type="CAUSAL_LM",
        ))
    if cfg_get(cfg.train, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required with LoRA + checkpointing
    return model, tok


def save_with_latest(model, out_dir) -> None:
    """save_pretrained + maintain an LATEST symlink (the hand-off contract)."""
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)
    latest = out_dir.parent / "LATEST"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.name)


# --------------------------------------------------------------------------- dumps

def dump_generations(model, tok, rows, out_path, *, run_id, model_name,
                     condition, max_new_tokens=64, limit=None) -> int:
    """Write §5 generations.jsonl — the only file the eval harness consumes."""
    from .prompting import format_prompt

    rows = rows[:limit] if limit else rows
    model.eval()
    n = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in rows:
            prompt = format_prompt(rec["history"], rec["profile_self"],
                                   rec.get("profile_partner"))
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            f.write(json.dumps({
                "run_id": run_id,
                "model": model_name,
                "condition": condition,
                "dialog_id": rec.get("id", str(n)),
                "profile_self": rec["profile_self"],
                "profile_partner": rec.get("profile_partner", []),
                "response": text.strip(),
            }, ensure_ascii=False) + "\n")
            n += 1
    return n
