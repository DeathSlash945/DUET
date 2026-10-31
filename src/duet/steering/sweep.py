"""Layer × magnitude sweep orchestration for DUET Component B.

This module owns steering intervention bookkeeping, not evaluation ownership.
Callers provide generation and optional scoring callbacks; the sweep records
their returned scalar metrics in JSON and CSV for the shared evaluation stack.
Results are never overwritten.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from duet.steering.apply import Direction, compose_directions, install_steering_hooks, read_direction


@dataclass(frozen=True)
class SweepPoint:
    """One single-layer steering condition."""

    layer: int
    magnitude: float
    attribute_ids: tuple[str, ...]


def sweep_points(
    *, layers: Iterable[int], magnitudes: Iterable[float], attribute_ids: Iterable[str]
) -> List[SweepPoint]:
    """Create a deterministic Cartesian layer × magnitude sweep plan."""
    attrs = tuple(sorted(set(attribute_ids)))
    layer_values = sorted(set(int(layer) for layer in layers))
    magnitude_values = list(magnitudes)
    if not attrs or not layer_values or not magnitude_values:
        raise ValueError("layers, magnitudes, and attribute_ids must all be non-empty.")
    if any(layer < 0 for layer in layer_values):
        raise ValueError("layers must be non-negative.")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in magnitude_values):
        raise ValueError("magnitudes must be numeric.")
    return [SweepPoint(layer, float(magnitude), attrs) for layer in layer_values for magnitude in magnitude_values]


def discover_consistent_directions(
    root: Union[str, Path],
    *,
    attribute_ids: Iterable[str],
    layers: Iterable[int],
    expected_model: str,
    min_pairs: int = 256,
) -> Dict[tuple[str, int], Direction]:
    """Load directions recursively, accepting retries only when bit-identical.

    The 256-pair extractor writes retry attempts to separate directories. A
    retry is safe to reuse only when its metadata and vector exactly match the
    earlier candidate; divergent duplicates fail loudly instead of silently
    selecting a result.
    """
    ids = set(attribute_ids)
    requested_layers = {int(layer) for layer in layers}
    needed = {(attr, layer) for attr in ids for layer in requested_layers}
    grouped: Dict[tuple[str, int], List[Direction]] = {}
    for path in sorted(Path(root).rglob("*.npz")):
        direction = read_direction(path)
        key = (direction.attribute_id, direction.layer)
        if key in needed:
            grouped.setdefault(key, []).append(direction)
    missing = sorted(needed - set(grouped))
    if missing:
        raise FileNotFoundError(f"Missing sweep directions: {missing}")
    selected: Dict[tuple[str, int], Direction] = {}
    for key, candidates in grouped.items():
        first = candidates[0]
        if (first.model_id != expected_model or first.dataset_version != "original" or
                first.profile_side != "self" or first.activation != "last_token" or
                first.n_pairs < min_pairs):
            raise ValueError(f"{key}: incompatible direction provenance.")
        for other in candidates[1:]:
            same_metadata = (
                other.model_id == first.model_id and other.dataset_version == first.dataset_version and
                other.profile_side == first.profile_side and other.activation == first.activation and
                other.n_pairs == first.n_pairs
            )
            if not same_metadata or not np.array_equal(other.vector, first.vector):
                raise ValueError(f"{key}: retry direction candidates differ; choose an explicit source.")
        selected[key] = first
    return selected


def directions_for_point(
    directions: Mapping[tuple[str, int], Direction], point: SweepPoint
) -> List[Direction]:
    """Select every requested attribute's direction for one intervention layer."""
    return [directions[(attribute_id, point.layer)] for attribute_id in point.attribute_ids]


def write_sweep_results(records: Sequence[Mapping[str, Any]], output_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Write JSON + scalar-friendly CSV, refusing to overwrite either file."""
    output_dir = Path(output_dir)
    json_path = output_dir / "sweep_results.json"
    csv_path = output_dir / "sweep_results.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep results in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path.write_text(json.dumps(list(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = sorted({key for record in records for key, value in record.items()
                     if isinstance(value, (str, int, float, bool)) or value is None})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def run_sweep(
    model,
    *,
    directions: Mapping[tuple[str, int], Direction],
    points: Sequence[SweepPoint],
    generate: Callable[[Any, SweepPoint], Sequence[Any]],
    score: Optional[Callable[[Sequence[Any], SweepPoint], Mapping[str, Union[int, float, str, bool]]]] = None,
    output_dir: Union[str, Path],
    position: str = "last_token",
) -> List[Dict[str, Any]]:
    """Run hooks and persist one record per point.

    ``generate`` owns decoding and returns its generated items. ``score`` is
    optional and must return scalar values only; this keeps metric definition
    and final evaluation in the evaluation workstream.
    """
    records: List[Dict[str, Any]] = []
    for point in points:
        selected = directions_for_point(directions, point)
        vectors = compose_directions(selected, magnitudes=point.magnitude)
        with install_steering_hooks(model, vectors, position=position):
            generated = list(generate(model, point))
        record: Dict[str, Any] = {
            **asdict(point),
            "attribute_ids": list(point.attribute_ids),
            "n_attributes": len(point.attribute_ids),
            "n_generated": len(generated),
            "position": position,
        }
        if score is not None:
            metrics = dict(score(generated, point))
            invalid = {key: value for key, value in metrics.items()
                       if not isinstance(value, (str, int, float, bool)) or isinstance(value, (list, dict, tuple))}
            if invalid:
                raise ValueError(f"score returned non-scalar metrics: {sorted(invalid)}")
            record.update(metrics)
        records.append(record)
    write_sweep_results(records, output_dir)
    return records
