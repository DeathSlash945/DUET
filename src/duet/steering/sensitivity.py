"""Persona-sensitivity definition: target log-prob under true vs perturbed profiles."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


@dataclass(frozen=True)
class SensitivityCase:
    id: str
    true_prompt: str
    perturbed_prompt: str
    target: str


def conditional_target_logprob(model, input_ids, *, target_start: int) -> float:
    """Sum causal log-probabilities of ``input_ids[:, target_start:]``.

    ``target_start`` is an explicit token boundary, avoiding an ambiguous
    profile/target text split during scoring. Inputs are batch-size one by
    design so no padding mask can leak into the metric definition.
    """
    import torch
    import torch.nn.functional as F

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length].")
    if not 1 <= target_start < input_ids.shape[1]:
        raise ValueError("target_start must leave at least one prompt and one target token.")
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs[:, target_start - 1:].sum().item())


def _tokenized_case(tokenizer, prompt: str, target: str, *, separator: str, device: str):
    import torch
    prompt_ids = tokenizer(prompt + separator, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not prompt_ids or not target_ids:
        raise ValueError("Prompt and target must tokenize to at least one token.")
    return torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device), len(prompt_ids)


def persona_sensitivity_delta(model, tokenizer, case: SensitivityCase, *, device: str = "cpu", separator: str = " ") -> Dict[str, Union[str, float, int]]:
    """Compute log P(target|true profile) - log P(target|perturbed profile)."""
    true_ids, true_start = _tokenized_case(tokenizer, case.true_prompt, case.target, separator=separator, device=device)
    perturbed_ids, perturbed_start = _tokenized_case(tokenizer, case.perturbed_prompt, case.target, separator=separator, device=device)
    true_logprob = conditional_target_logprob(model, true_ids, target_start=true_start)
    perturbed_logprob = conditional_target_logprob(model, perturbed_ids, target_start=perturbed_start)
    return {"id": case.id, "true_logprob": true_logprob, "perturbed_logprob": perturbed_logprob,
            "logprob_delta": true_logprob - perturbed_logprob,
            "true_target_tokens": int(true_ids.shape[1] - true_start),
            "perturbed_target_tokens": int(perturbed_ids.shape[1] - perturbed_start)}


def run_sensitivity_experiment(model, tokenizer, cases: Iterable[SensitivityCase], *, output_dir: Union[str, Path], device: str = "cpu") -> List[Dict[str, Union[str, float, int]]]:
    """Compute and save the metric definition's per-example results separately."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite sensitivity results: {output_dir}")
    records = [persona_sensitivity_delta(model, tokenizer, case, device=device) for case in cases]
    if not records:
        raise ValueError("At least one sensitivity case is required.")
    output_dir.mkdir(parents=True)
    (output_dir / "sensitivity_results.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "sensitivity_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records
