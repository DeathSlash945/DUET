"""Convert PAL-format PERSONA-CHAT (the recovered baseline files) into the
DUET §5 dialogue contract, so configs/sft.yaml and configs/dpo_*.yaml run
unmodified.

Limitation by design: PAL's flat format only carries the SPEAKER's own
profile — profile_partner comes back empty. The full both-profile record is
produced by Zala's pipeline (scripts/01) from raw PERSONA-CHAT; this converter
is the Phase-0 stand-in for baseline runs.

Usage:
    python -m duet.baselines.pal.to_contract \
        --src data/raw/pal_format --out data/processed
"""

import argparse
import json
from pathlib import Path

from duet.training.common import write_jsonl

# PAL file name -> contract split name
SPLITS = {
    "train_origin.txt": "train.jsonl",
    "valid_origin.txt": "valid.jsonl",
    "test.txt": "test.jsonl",
}


def convert_row(ex, idx: int, split: str) -> dict:
    persona, history, target, labels = ex[0], ex[1], ex[2], ex[3]
    return {
        "id": f"{split}_{idx:06d}",
        "history": list(history),
        "profile_self": list(persona),
        "profile_partner": [],          # not recoverable from PAL format
        "target": target,
        "persona_labels": list(labels),
        "dataset_version": "origin",
    }


def convert_dir(src: Path, out: Path) -> dict:
    stats = {}
    for pal_name, split_name in SPLITS.items():
        path = src / pal_name
        if not path.exists():
            continue
        rows = json.load(open(path, encoding="utf-8"))
        write_jsonl(out / split_name, (convert_row(r, i, split_name.split(".")[0])
                                       for i, r in enumerate(rows)))
        stats[split_name] = len(rows)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/raw/pal_format")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    stats = convert_dir(Path(a.src), Path(a.out))
    if not stats:
        raise SystemExit(f"no PAL files found in {a.src}")
    for name, n in stats.items():
        print(f"{name}: {n} rows")


if __name__ == "__main__":
    main()
