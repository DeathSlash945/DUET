"""
Optional integration checks against the real downloaded PERSONA-CHAT data.

Skipped automatically when data/processed/ has not been built yet (CI / fresh
clones). Build with scripts/00_download_data.sh + scripts/01_build_dataset.sh.

The expected record counts were OBSERVED from the official SHA256-pinned
personachat.tgz (ParlAI build.py checksum) on 2026-08-21 and act as a
regression baseline for the fbdialog parser.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from duet.data.schemas import ProcessedRecord

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ("train", "valid", "test")
VERSIONS = ("original", "revised")

# Observed from the official archive (episodes: 8939 / 1000 / 968 per version).
EXPECTED_RECORDS = {
    ("original", "train"): 65719,
    ("original", "valid"): 7801,
    ("original", "test"): 7512,
    ("revised", "train"): 65719,
    ("revised", "valid"): 7801,
    ("revised", "test"): 7512,
}


def _load(version, split):
    path = PROCESSED / version / f"{split}.jsonl"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ProcessedRecord.from_dict(json.loads(line)) for line in f]


@unittest.skipUnless(
    (PROCESSED / "original" / "train.jsonl").exists(),
    "real processed data not built (run scripts/01_build_dataset.sh)",
)
class RealDataContractTest(unittest.TestCase):

    def test_counts_records_validate_and_ids_unique(self):
        for version in VERSIONS:
            for split in SPLITS:
                records = _load(version, split)
                self.assertIsNotNone(records, f"missing {version}/{split}.jsonl")
                self.assertEqual(
                    len(records), EXPECTED_RECORDS[(version, split)],
                    f"unexpected record count for {version}/{split}",
                )
                ids = set()
                for rec in records:
                    # both-profiles-present invariant on every real record
                    self.assertTrue(rec.profile_self, f"{version}/{split}: {rec.id}")
                    self.assertTrue(rec.profile_partner, f"{version}/{split}: {rec.id}")
                    self.assertEqual(rec.dataset_version, version, rec.id)
                    ids.add(rec.id)
                self.assertEqual(len(ids), len(records), f"duplicate ids in {version}/{split}")

    def test_original_revised_share_dialogues_but_differ_in_profiles(self):
        for split in SPLITS:
            original = _load("original", split)
            revised = _load("revised", split)
            self.assertEqual(
                len(original), len(revised),
                "Original and Revised must keep the same dialogue structure",
            )
            # Revised rephrases personas -> profiles differ ...
            profile_diffs = sum(
                1 for a, b in zip(original, revised)
                if a.profile_self != b.profile_self
                or a.profile_partner != b.profile_partner
            )
            self.assertGreater(profile_diffs, 0, f"no revised profiles differ in {split}")
            # ... while the dialogues themselves stay aligned turn for turn.
            dialogue_matches = sum(
                1 for a, b in zip(original, revised)
                if a.history == b.history and a.target == b.target
            )
            self.assertEqual(
                dialogue_matches, len(original),
                f"dialogues diverge between versions in {split}",
            )


if __name__ == "__main__":
    unittest.main()
