"""Inference-time residual-stream steering hooks for DUET Component B.

Directions are the raw mean differences produced by ``steering.extract``.
This module normalizes every nonzero direction to unit L2 norm before applying
the configured signed magnitude, so magnitudes are comparable across layers
and attributes.  Multiple attributes compose additively at each layer.

The default ``last_token`` mode changes only ``hidden[:, -1, :]`` in each
hook invocation.  It is therefore intended for batch-size-one autoregressive
generation, including the prompt pass and every one-token decode pass.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Union

import numpy as np

from duet.steering.extract import load_direction


Position = Literal["last_token", "all_tokens"]


@dataclass(frozen=True)
class Direction:
    """Validated direction and provenance loaded from an extraction NPZ."""

    attribute_id: str
    canonical: str
    layer: int
    vector: np.ndarray
    n_pairs: int
    model_id: str
    dataset_version: str
    profile_side: str
    activation: str


def _scalar(data: Mapping[str, np.ndarray], key: str) -> object:
    if key not in data:
        raise ValueError(f"Direction NPZ is missing metadata key {key!r}.")
    return data[key].item()


def read_direction(path: Union[str, Path]) -> Direction:
    """Load and validate one direction file without modifying it."""
    path = Path(path)
    data = load_direction(path)
    vector = np.asarray(data.get("direction"), dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{path}: direction must be a non-empty 1-D vector.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{path}: direction contains non-finite values.")
    layer = int(_scalar(data, "layer"))
    n_pairs = int(_scalar(data, "n_pairs"))
    if layer < 0 or n_pairs < 1:
        raise ValueError(f"{path}: layer must be >= 0 and n_pairs must be >= 1.")
    return Direction(
        attribute_id=str(_scalar(data, "attribute_id")),
        canonical=str(_scalar(data, "canonical")),
        layer=layer,
        vector=vector,
        n_pairs=n_pairs,
        model_id=str(_scalar(data, "model_id")),
        dataset_version=str(_scalar(data, "dataset_version")),
        profile_side=str(_scalar(data, "profile_side")),
        activation=str(_scalar(data, "activation")),
    )


def discover_directions(
    root: Union[str, Path],
    *,
    attribute_ids: Iterable[str],
    layers: Iterable[int],
    expected_model: Optional[str] = None,
    expected_version: str = "original",
    expected_side: str = "self",
) -> Dict[tuple[str, int], Direction]:
    """Find exactly one validated NPZ per requested attribute/layer.

    Recursive discovery accommodates the resumable extraction's chunked
    output layout. Duplicate candidates are rejected rather than silently
    choosing one retry output.
    """
    root = Path(root)
    requested_ids = set(attribute_ids)
    requested_layers = {int(layer) for layer in layers}
    if not requested_ids or not requested_layers:
        raise ValueError("attribute_ids and layers must both be non-empty.")
    found: Dict[tuple[str, int], Direction] = {}
    for path in sorted(root.rglob("*.npz")):
        direction = read_direction(path)
        key = (direction.attribute_id, direction.layer)
        if key not in {(a, layer) for a in requested_ids for layer in requested_layers}:
            continue
        if direction.dataset_version != expected_version or direction.profile_side != expected_side:
            raise ValueError(f"{path}: incompatible dataset version or profile side.")
        if expected_model is not None and direction.model_id != expected_model:
            raise ValueError(f"{path}: expected model {expected_model!r}, got {direction.model_id!r}.")
        if key in found:
            raise ValueError(f"Duplicate direction candidates for {key}: {path}")
        found[key] = direction
    missing = sorted((a, layer) for a in requested_ids for layer in requested_layers if (a, layer) not in found)
    if missing:
        raise FileNotFoundError(f"Missing direction files for: {missing}")
    return found


def compose_directions(
    directions: Iterable[Direction],
    *,
    magnitudes: Union[float, Mapping[str, float]],
) -> Dict[int, np.ndarray]:
    """Return L2-normalized, magnitude-weighted summed vectors by layer."""
    result: Dict[int, np.ndarray] = {}
    for direction in directions:
        magnitude = magnitudes.get(direction.attribute_id) if isinstance(magnitudes, Mapping) else magnitudes
        if not isinstance(magnitude, (int, float)) or isinstance(magnitude, bool):
            raise ValueError(f"Magnitude for {direction.attribute_id!r} must be numeric.")
        norm = float(np.linalg.norm(direction.vector))
        if norm == 0:
            raise ValueError(f"Cannot steer with zero-norm direction: {direction.attribute_id!r}, layer {direction.layer}.")
        contribution = (float(magnitude) * direction.vector / norm).astype(np.float32)
        if direction.layer in result:
            if result[direction.layer].shape != contribution.shape:
                raise ValueError(f"Layer {direction.layer} has incompatible direction dimensions.")
            result[direction.layer] += contribution
        else:
            result[direction.layer] = contribution
    if not result:
        raise ValueError("At least one direction is required.")
    return result


def transformer_blocks(model) -> Sequence:
    """Return common Hugging Face causal-LM transformer block containers."""
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),  # Qwen/Llama
        getattr(getattr(model, "transformer", None), "h", None),  # GPT-2
    )
    for blocks in candidates:
        if blocks is not None:
            return blocks
    raise ValueError("Cannot locate transformer blocks; expected model.model.layers or model.transformer.h.")


class SteeringHooks(AbstractContextManager):
    """A removable set of forward hooks that adds residual steering vectors."""

    def __init__(self, model, vectors_by_layer: Mapping[int, np.ndarray], *, position: Position = "last_token"):
        if position not in ("last_token", "all_tokens"):
            raise ValueError("position must be 'last_token' or 'all_tokens'.")
        self._position = position
        self._handles = []
        blocks = transformer_blocks(model)
        for layer, vector in vectors_by_layer.items():
            if not isinstance(layer, int) or layer < 0 or layer >= len(blocks):
                raise ValueError(f"Layer {layer} is outside model block range [0, {len(blocks)}).")
            self._handles.append(blocks[layer].register_forward_hook(self._hook(vector)))

    def _hook(self, vector: np.ndarray):
        def apply(_module, _inputs, output):
            import torch

            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[-1] != vector.shape[0]:
                raise ValueError(f"Direction dimension {vector.shape[0]} != hidden size {hidden.shape[-1]}.")
            delta = torch.as_tensor(vector, dtype=hidden.dtype, device=hidden.device)
            changed = hidden.clone()
            if self._position == "last_token":
                changed[:, -1, :] += delta
            else:
                changed += delta.view(1, 1, -1)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed

        return apply

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def install_steering_hooks(model, vectors_by_layer: Mapping[int, np.ndarray], *, position: Position = "last_token") -> SteeringHooks:
    """Install residual hooks; call ``close()`` or use a ``with`` block."""
    return SteeringHooks(model, vectors_by_layer, position=position)
