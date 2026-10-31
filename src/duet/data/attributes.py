"""
Profile attribute segmentation for DUET Component B.

Turns PERSONA-CHAT profiles (lists of persona sentences) into individual
attribute segments and builds a corpus-level inventory that tells Component B
which attributes have enough contexts for direction extraction.

Design decisions (documented per AGENTS.md verification expectations):

* Segmentation unit — sentence level by default: each non-empty sentence of a
  profile entry becomes one segment (PERSONA-CHAT personas are crowd-written
  single-fact sentences). Optional ``split_compound=True`` additionally splits
  coordinating-conjunction clauses ("i like x and i like y") when BOTH sides
  look like independent clauses (leading pronoun/auxiliary/common persona verb,
  >= 2 tokens). Elliptical coordination ("i like cats and dogs") is never
  split, because the second fragment is not a standalone attribute.
* canonical — grouping key: lowercase, all non-alphanumeric runs collapsed to
  single spaces. Verbatim ``text`` is always preserved alongside it.
* attribute_id — filename-safe and collision-free:
  ``{slug(canonical, <=48 chars)}_{sha1(canonical)[:8]}``. Safe for the
  contract path ``data/attrs/{layer}/{attribute}.npz`` (project_plan.md §5.3).
* Count semantics in the inventory:
    count     — occurrences across record-profiles (record-weighted: a profile
                shared by its episode's ~9 records counts once per record),
                i.e. how many training contexts express the attribute.
    n_profiles — distinct profiles containing the attribute at least once.
* Dataset versions are inventoried separately (never merged), per the
  pipeline rule to verify Original and Revised separately.

CLI (config-driven, see configs/attributes.yaml):
    python -m duet.data.attributes --config configs/attributes.yaml
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from duet.data.schemas import ProcessedRecord, DatasetVersion

# ---------------------------------------------------------------------------
# Segmentation primitives
# ---------------------------------------------------------------------------

# Split a profile entry into sentences. Handles both PERSONA-CHAT punctuation
# styles ("done." and "done ."); never splits without following whitespace, so
# token-internal dots stay intact.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_COMPOUND_SPLIT_RE = re.compile(r"\s+(?:and|but)\s+", re.IGNORECASE)

# Leading tokens that mark an independent clause for compound splitting.
# Includes subject/auxiliary pronouns and common PERSONA-CHAT persona verbs.
_CLAUSE_STARTERS = frozenset({
    "i", "im", "me", "my", "mine", "we", "our", "us", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "it", "its",
    "am", "is", "are", "was", "were", "do", "does", "did", "have", "has",
    "had", "can", "could", "will", "would", "should", "must",
    "like", "love", "enjoy", "hate", "play", "read", "watch", "eat", "drink",
    "cook", "run", "swim", "ski", "hike", "dance", "sing", "write", "draw",
    "work", "live", "study", "travel", "listen", "visit", "collect", "train",
    "ride", "drive", "fly", "go", "make", "build", "fix", "paint", "code",
})


def canonicalize(text: str) -> str:
    """Normalized grouping key: lowercase, non-alphanumeric runs collapsed."""
    if not isinstance(text, str):
        raise ValueError("canonicalize() expects a string.")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def attribute_id(canonical: str) -> str:
    """Stable filename-safe id: ``{slug}_{sha1[:8]}`` of the canonical key."""
    slug = re.sub(r"[^a-z0-9]+", "_", canonical).strip("_")
    if len(slug) > 48:
        slug = slug[:48].rstrip("_")
    suffix = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{suffix}" if slug else suffix


def split_sentences(text: str) -> List[str]:
    """Split one profile entry into its sentences (empty pieces dropped)."""
    if not isinstance(text, str):
        raise ValueError("split_sentences() expects a string.")
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _is_clause(fragment: str) -> bool:
    tokens = fragment.lower().split()
    return len(tokens) >= 2 and tokens[0] in _CLAUSE_STARTERS


def split_compound_sentence(sentence: str) -> List[str]:
    """Split 'i like x and i like y' style clauses; never elliptical forms.

    Recursion splits at the first conjunction whose BOTH sides look like
    independent clauses; if no such conjunction exists the sentence is
    returned unchanged (conservative by design).
    """
    for match in _COMPOUND_SPLIT_RE.finditer(sentence):
        left = sentence[:match.start()]
        right = sentence[match.end():]
        if _is_clause(left) and _is_clause(right):
            return (split_compound_sentence(left) + split_compound_sentence(right))
    return [sentence]


@dataclass
class AttributeSegment:
    """One individual persona attribute carved out of a profile.

    Attributes:
        attribute_id: Stable filename-safe identifier (see attribute_id()).
        text: Verbatim segment text as it appears in the profile.
        canonical: Normalized grouping key (see canonicalize()).
        role: Which speaker's profile the segment came from.
        order: 0-based position of the segment within that profile.
    """
    attribute_id: str
    text: str
    canonical: str
    role: str
    order: int

    def __post_init__(self) -> None:
        if not isinstance(self.attribute_id, str) or not self.attribute_id.strip():
            raise ValueError("Field 'attribute_id' must be a non-empty string.")
        if not re.fullmatch(r"[a-z0-9_]+", self.attribute_id):
            raise ValueError(
                f"Field 'attribute_id' must be filename-safe [a-z0-9_]: {self.attribute_id!r}"
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Field 'text' must be a non-empty string.")
        if not isinstance(self.canonical, str) or not self.canonical.strip():
            raise ValueError("Field 'canonical' must be a non-empty string.")
        if self.role not in ("self", "partner"):
            raise ValueError("Field 'role' must be 'self' or 'partner'.")
        if not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 0:
            raise ValueError("Field 'order' must be a non-negative integer.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeSegment":
        if not isinstance(data, dict):
            raise TypeError("AttributeSegment.from_dict() expects a dictionary.")
        required = ["attribute_id", "text", "canonical", "role", "order"]
        missing = [f for f in required if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in AttributeSegment: {missing}")
        return cls(**{f: data[f] for f in required})


def segment_profile(
    profile: List[str],
    *,
    role: str,
    split_compound: bool = False,
) -> List[AttributeSegment]:
    """Segment one speaker's profile into individual attribute segments.

    Args:
        profile: List of persona sentences (ProcessedRecord.profile_* field).
        role: 'self' or 'partner' (which speaker owns the profile).
        split_compound: Also split conjunction-joined clauses (see module
            docstring); default False keeps segmentation verbatim.

    Returns:
        Segments in profile order. Entries that yield no non-empty canonical
        form (e.g. punctuation-only) are dropped, never crash the build.
    """
    if not isinstance(profile, (list, tuple)) or not all(isinstance(p, str) for p in profile):
        raise ValueError("profile must be a list/tuple of strings.")
    if role not in ("self", "partner"):
        raise ValueError("role must be 'self' or 'partner'.")

    segments: List[AttributeSegment] = []
    for entry in profile:
        pieces = split_sentences(entry)
        if split_compound:
            pieces = [part for p in pieces for part in split_compound_sentence(p)]
        for piece in pieces:
            canonical = canonicalize(piece)
            if not canonical:
                continue
            segments.append(
                AttributeSegment(
                    attribute_id=attribute_id(canonical),
                    text=piece,
                    canonical=canonical,
                    role=role,
                    order=len(segments),
                )
            )
    return segments


def segment_record(
    record: ProcessedRecord,
    *,
    split_compound: bool = False,
) -> Dict[str, List[AttributeSegment]]:
    """Segment BOTH profiles of a processed record ('self' and 'partner')."""
    if not isinstance(record, ProcessedRecord):
        raise TypeError("segment_record() expects a ProcessedRecord.")
    return {
        "self": segment_profile(record.profile_self, role="self", split_compound=split_compound),
        "partner": segment_profile(record.profile_partner, role="partner", split_compound=split_compound),
    }


# ---------------------------------------------------------------------------
# Corpus inventory
# ---------------------------------------------------------------------------

@dataclass
class AttributeStats:
    """Frequency of one attribute across the inventoried records."""
    attribute_id: str
    canonical: str
    count: int
    n_profiles: int
    n_self: int
    n_partner: int
    examples: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.attribute_id, str) or not self.attribute_id.strip():
            raise ValueError("Field 'attribute_id' must be a non-empty string.")
        if not isinstance(self.canonical, str) or not self.canonical.strip():
            raise ValueError("Field 'canonical' must be a non-empty string.")
        for name in ("count", "n_profiles", "n_self", "n_partner"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Field '{name}' must be a non-negative integer.")
        if not isinstance(self.examples, list) or not all(isinstance(e, str) for e in self.examples):
            raise ValueError("Field 'examples' must be a list of strings.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeStats":
        if not isinstance(data, dict):
            raise TypeError("AttributeStats.from_dict() expects a dictionary.")
        required = ["attribute_id", "canonical", "count", "n_profiles", "n_self", "n_partner"]
        missing = [f for f in required if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in AttributeStats: {missing}")
        return cls(examples=list(data.get("examples", [])), **{f: data[f] for f in required})


@dataclass
class AttributeInventory:
    """All attributes of one dataset version with frequencies and parameters.

    Attributes:
        dataset_version: 'original' | 'revised' | 'mixed' (never merged by the
            CLI — versions are always inventoried separately).
        params: Segmentation/build parameters used (reproducibility record).
        totals: n_records, n_segments, n_unique (pre-filter), n_kept.
        attributes: Stats sorted by count desc, then canonical asc.
    """
    dataset_version: str
    params: Dict[str, Any]
    totals: Dict[str, int]
    attributes: List[AttributeStats]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_version, str) or not self.dataset_version.strip():
            raise ValueError("Field 'dataset_version' must be a non-empty string.")
        if not isinstance(self.params, dict):
            raise ValueError("Field 'params' must be a dictionary.")
        if not isinstance(self.totals, dict):
            raise ValueError("Field 'totals' must be a dictionary.")
        if not isinstance(self.attributes, list) or not all(
            isinstance(a, AttributeStats) for a in self.attributes
        ):
            raise ValueError("Field 'attributes' must be a list of AttributeStats.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "params": self.params,
            "totals": self.totals,
            "attributes": [a.to_dict() for a in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeInventory":
        if not isinstance(data, dict):
            raise TypeError("AttributeInventory.from_dict() expects a dictionary.")
        required = ["dataset_version", "params", "totals", "attributes"]
        missing = [f for f in required if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in AttributeInventory: {missing}")
        return cls(
            dataset_version=data["dataset_version"],
            params=dict(data["params"]),
            totals=dict(data["totals"]),
            attributes=[AttributeStats.from_dict(a) for a in data["attributes"]],
        )


def _profile_key(profile: List[str]) -> str:
    """Distinct-profile identity: hash of the sentence tuple."""
    return hashlib.sha1("\x1f".join(profile).encode("utf-8")).hexdigest()[:12]


def build_attribute_inventory(
    records: Iterable[ProcessedRecord],
    *,
    dataset_version: Optional[DatasetVersion] = None,
    split_compound: bool = False,
    min_count: int = 1,
    max_examples: int = 5,
    params: Optional[Dict[str, Any]] = None,
) -> AttributeInventory:
    """Aggregate attribute frequencies over processed records.

    Args:
        records: ProcessedRecord instances (both profiles are inventoried).
        dataset_version: Enforced version; None infers it (one version =>
            that version, several => 'mixed'). A record whose version differs
            from the enforced value raises — catches wrong-directory wiring.
        split_compound: Passed through to segment_profile().
        min_count: Drop attributes whose record-weighted count is below this.
        max_examples: Distinct verbatim surface forms kept per attribute.
        params: Extra build parameters to embed in the inventory (CLI passes
            splits, config path, etc.).
    """
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be an integer >= 1.")
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 0:
        raise ValueError("max_examples must be a non-negative integer.")

    recs = list(records)
    versions = {getattr(r, "dataset_version", None) for r in recs}
    if dataset_version is None:
        dataset_version = versions.pop() if len(versions) == 1 else "mixed"
    elif versions - {dataset_version}:
        raise ValueError(
            f"dataset_version={dataset_version!r} enforced but records contain "
            f"{sorted(v for v in versions if v != dataset_version)}"
        )

    count: Dict[str, int] = {}
    n_self: Dict[str, int] = {}
    n_partner: Dict[str, int] = {}
    profile_keys: Dict[str, set] = {}
    surfaces: Dict[str, List[str]] = {}
    canonical_by_id: Dict[str, str] = {}
    # Same profile repeats across its episode's records — segment it once.
    segment_cache: Dict[Tuple[str, ...], List[AttributeSegment]] = {}

    n_segments = 0
    for record in recs:
        for role, profile in (("self", record.profile_self), ("partner", record.profile_partner)):
            # Role is part of the key: the same persona list can serve as 'self'
            # in one episode and as 'partner' in another — segments must keep
            # the role they were segmented for.
            cache_key = (role,) + tuple(profile)
            segments = segment_cache.get(cache_key)
            if segments is None:
                segments = segment_profile(
                    profile, role=role, split_compound=split_compound
                )
                segment_cache[cache_key] = segments
            if not segments:
                continue
            pkey = _profile_key(profile)
            for seg in segments:
                n_segments += 1
                count[seg.attribute_id] = count.get(seg.attribute_id, 0) + 1
                if role == "self":
                    n_self[seg.attribute_id] = n_self.get(seg.attribute_id, 0) + 1
                else:
                    n_partner[seg.attribute_id] = n_partner.get(seg.attribute_id, 0) + 1
                canonical_by_id[seg.attribute_id] = seg.canonical
                profile_keys.setdefault(seg.attribute_id, set()).add(pkey)
                seen = surfaces.setdefault(seg.attribute_id, [])
                if seg.text not in seen and len(seen) < max_examples:
                    seen.append(seg.text)

    stats = [
        AttributeStats(
            attribute_id=aid,
            canonical=canonical_by_id[aid],
            count=count[aid],
            n_profiles=len(profile_keys[aid]),
            n_self=n_self.get(aid, 0),
            n_partner=n_partner.get(aid, 0),
            examples=surfaces.get(aid, []),
        )
        for aid in count
        if count[aid] >= min_count
    ]
    stats.sort(key=lambda s: (-s.count, s.canonical))

    build_params = dict(params or {})
    build_params.update(
        split_compound=split_compound,
        min_count=min_count,
        max_examples=max_examples,
    )
    return AttributeInventory(
        dataset_version=dataset_version,
        params=build_params,
        totals={
            "n_records": len(recs),
            "n_segments": n_segments,
            "n_unique": len(count),
            "n_kept": len(stats),
        },
        attributes=stats,
    )


def save_inventory(inventory: AttributeInventory, path: Union[str, Path]) -> str:
    """Write the inventory as compact JSON (deterministic, no timestamps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inventory.to_dict(), f, ensure_ascii=False)
        f.write("\n")
    return str(path)


def load_inventory(path: Union[str, Path]) -> AttributeInventory:
    """Load and validate an inventory JSON written by save_inventory()."""
    with open(path, "r", encoding="utf-8") as f:
        return AttributeInventory.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BUILTIN_DEFAULTS = {
    "processed_dir": "data/processed",
    "output_dir": "data/attrs",
    "versions": ["original", "revised"],
    "splits": ["train"],
    "split_compound": False,
    "min_count": 1,
    "max_examples": 5,
}


def _load_records(processed_dir: Path, version: str, split: str) -> List[ProcessedRecord]:
    path = processed_dir / version / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — build processed data first (scripts/01_build_dataset.sh)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [ProcessedRecord.from_dict(json.loads(line)) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment PERSONA-CHAT profiles into attributes and build inventories."
    )
    parser.add_argument("--config", default="configs/attributes.yaml",
                        help="YAML config (CLI flags override it; missing file -> built-in defaults).")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--versions", nargs="+", default=None,
                        choices=["original", "revised"])
    parser.add_argument("--splits", nargs="+", default=None,
                        choices=["train", "valid", "test"])
    parser.add_argument("--split-compound", action="store_true", default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = dict(_BUILTIN_DEFAULTS)
    config_path = Path(args.config)
    if config_path.exists():
        import yaml  # PyYAML (already present via the datasets dependency)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        unknown = set(loaded) - set(_BUILTIN_DEFAULTS)
        if unknown:
            raise KeyError(f"Unknown key(s) in {config_path}: {sorted(unknown)}")
        cfg.update(loaded)
        print(f"[attributes] config: {config_path}")
    else:
        print(f"[attributes] config not found ({config_path}); using built-in defaults")

    for key in ("processed_dir", "output_dir", "versions", "splits",
                "split_compound", "min_count", "max_examples"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    processed_dir = Path(cfg["processed_dir"])
    output_dir = Path(cfg["output_dir"])
    top_n = 10

    for version in cfg["versions"]:
        records: List[ProcessedRecord] = []
        for split in cfg["splits"]:
            part = _load_records(processed_dir, version, split)
            print(f"[attributes] {version}/{split}: {len(part)} records")
            records.extend(part)

        inventory = build_attribute_inventory(
            records,
            dataset_version=version,
            split_compound=cfg["split_compound"],
            min_count=cfg["min_count"],
            max_examples=cfg["max_examples"],
            params={
                "splits": list(cfg["splits"]),
                "config": str(config_path) if config_path.exists() else None,
            },
        )
        out_path = save_inventory(inventory, output_dir / f"inventory_{version}.json")
        totals = inventory.totals
        print(
            f"[attributes] {version}: records={totals['n_records']} "
            f"segments={totals['n_segments']} unique={totals['n_unique']} "
            f"kept={totals['n_kept']} -> {out_path}"
        )
        for stat in inventory.attributes[:top_n]:
            print(f"[attributes]   {stat.count:>7}  {stat.canonical}")


if __name__ == "__main__":
    main()
