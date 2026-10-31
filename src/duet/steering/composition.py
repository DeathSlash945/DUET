"""Multi-attribute and conflicting-pair Component B steering experiments."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from duet.steering.apply import Direction, compose_directions, install_steering_hooks


@dataclass(frozen=True)
class CompositionPoint:
    layer: int
    magnitude: float
    attribute_ids: tuple[str, ...]
    condition: str  # "composition" | "conflict"


def composition_sets(
    attribute_ids: Iterable[str], *, sizes: Iterable[int] = (2, 3),
    conflicting_pair: Optional[Sequence[str]] = None, max_sets_per_size: int = 0,
) -> List[tuple[tuple[str, ...], str]]:
    """Build deterministic 2/3-attribute sets plus a marked conflict pair.

    ``max_sets_per_size=0`` means all combinations; use a positive cap for a
    tractable experimental subset.  The conflict pair is included even when it
    is outside that cap and is never silently treated as an ordinary mixture.
    """
    ids = tuple(sorted(set(attribute_ids)))
    if not ids:
        raise ValueError("attribute_ids must be non-empty.")
    results: List[tuple[tuple[str, ...], str]] = []
    for size in sorted(set(int(value) for value in sizes)):
        if size not in (2, 3):
            raise ValueError("composition sizes must be 2 and/or 3.")
        combos = list(itertools.combinations(ids, size))
        if max_sets_per_size > 0:
            combos = combos[:max_sets_per_size]
        results.extend((combo, "composition") for combo in combos)
    if conflicting_pair is not None:
        conflict = tuple(sorted(conflicting_pair))
        if len(conflict) != 2 or len(set(conflict)) != 2 or not set(conflict).issubset(ids):
            raise ValueError("conflicting_pair must contain two distinct requested attribute IDs.")
        if (conflict, "conflict") not in results:
            results.append((conflict, "conflict"))
    return results


def composition_points(*, layers: Iterable[int], magnitudes: Iterable[float], sets: Iterable[tuple[tuple[str, ...], str]]) -> List[CompositionPoint]:
    """Expand composition sets into layer × magnitude conditions."""
    points = []
    for attributes, condition in sets:
        for layer in sorted(set(int(layer) for layer in layers)):
            for magnitude in magnitudes:
                points.append(CompositionPoint(layer, float(magnitude), attributes, condition))
    if not points:
        raise ValueError("No composition points were produced.")
    return points


def write_composition_results(records: Sequence[Mapping[str, Any]], output_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Persist composition records without overwriting prior experiments."""
    output_dir = Path(output_dir)
    json_path, csv_path = output_dir / "composition_results.json", output_dir / "composition_results.csv"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite composition results: {output_dir}")
    output_dir.mkdir(parents=True)
    json_path.write_text(json.dumps(list(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = sorted({key for record in records for key, value in record.items()
                     if isinstance(value, (str, int, float, bool)) or value is None})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def run_composition_experiment(model, *, directions: Mapping[tuple[str, int], Direction], points: Sequence[CompositionPoint],
                               generate: Callable[[Any, CompositionPoint], Sequence[Any]], output_dir: Union[str, Path],
                               score: Optional[Callable[[Sequence[Any], CompositionPoint], Mapping[str, Union[int, float, str, bool]]]] = None,
                               position: str = "last_token") -> List[Dict[str, Any]]:
    """Run composed hooks; generation/scoring stay caller-owned."""
    records: List[Dict[str, Any]] = []
    for point in points:
        selected = [directions[(attribute, point.layer)] for attribute in point.attribute_ids]
        vectors = compose_directions(selected, magnitudes=point.magnitude)
        with install_steering_hooks(model, vectors, position=position):
            generated = list(generate(model, point))
        record: Dict[str, Any] = {**asdict(point), "attribute_ids": list(point.attribute_ids),
                                  "n_attributes": len(point.attribute_ids), "n_generated": len(generated),
                                  "position": position}
        if score is not None:
            metrics = dict(score(generated, point))
            if any(not isinstance(value, (str, int, float, bool)) for value in metrics.values()):
                raise ValueError("Composition score must return scalar metrics only.")
            record.update(metrics)
        records.append(record)
    write_composition_results(records, output_dir)
    return records
