"""
Tests for src/duet/data/attributes.py — profile attribute segmentation.

Unit tests run on synthetic fixtures (no download needed). The real-data
tests build inventories from data/processed/ and auto-skip on fresh clones.
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from duet.data.attributes import (
    AttributeInventory,
    AttributeSegment,
    AttributeStats,
    attribute_id,
    build_attribute_inventory,
    canonicalize,
    load_inventory,
    save_inventory,
    segment_profile,
    segment_record,
    split_compound_sentence,
    split_sentences,
)
from duet.data.schemas import ProcessedRecord

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Observed from the official SHA256-pinned archive (train split).
EXPECTED_TRAIN_RECORDS = 65719

# Observed with configs/attributes.yaml params (sentence-level, min_count=1),
# built from the official archive on 2026-08-21 — regression baseline for the
# segmentation pipeline (deterministic build).
EXPECTED_INVENTORY = {
    "original": {"n_segments": 591035, "n_unique": 4725},
    "revised": {"n_segments": 590523, "n_unique": 4489},
}


def _record(version="original", self_profile=None, partner_profile=None, idx=0):
    return ProcessedRecord(
        id=f"{version}_train_ep{idx:05d}_t00",
        history=["hello !"],
        profile_self=self_profile if self_profile is not None else ["i like tea ."],
        profile_partner=partner_profile if partner_profile is not None else ["i love coffee ."],
        target="hi !",
        persona_labels=[],
        dataset_version=version,
    )


class SegmentationUnitTest(unittest.TestCase):

    def test_canonicalize_collapses_case_and_punctuation(self):
        self.assertEqual(canonicalize("I like TV,  and movies!"), "i like tv and movies")
        self.assertEqual(canonicalize("  i am 5 feet . tall?? "), "i am 5 feet tall")

    def test_attribute_id_stable_and_filename_safe(self):
        a = attribute_id("i like dogs")
        self.assertEqual(a, attribute_id("i like dogs"))       # stable
        self.assertNotEqual(a, attribute_id("i like cats"))    # distinct
        self.assertRegex(a, r"^[a-z0-9_]+$")                   # npz-safe
        self.assertTrue(a.startswith("i_like_dogs_"))

    def test_split_sentences_handles_both_punctuation_styles(self):
        self.assertEqual(
            split_sentences("i read books. i play soccer !  and i swim."),
            ["i read books.", "i play soccer !", "and i swim."],
        )
        self.assertEqual(split_sentences("   "), [])

    def test_segment_profile_sentence_level_default(self):
        segments = segment_profile(
            ["i read books. i play soccer !", "i love hiking ."],
            role="self",
        )
        self.assertEqual(
            [s.text for s in segments],
            ["i read books.", "i play soccer !", "i love hiking ."],
        )
        self.assertEqual([s.order for s in segments], [0, 1, 2])
        self.assertEqual({s.role for s in segments}, {"self"})
        self.assertEqual(segments[0].canonical, "i read books")

    def test_segment_profile_drops_empty_but_keeps_valid(self):
        segments = segment_profile(["...", "i like cats ."], role="partner")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "i like cats .")
        self.assertEqual(segments[0].role, "partner")

    def test_segment_profile_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            segment_profile(["ok", 42], role="self")
        with self.assertRaises(ValueError):
            segment_profile(["ok"], role="wrong")

    def test_compound_split_only_when_both_sides_are_clauses(self):
        # both sides independent -> split
        self.assertEqual(
            split_compound_sentence("i like cats and i like dogs"),
            ["i like cats", "i like dogs"],
        )
        # elliptical object coordination -> never split
        self.assertEqual(split_compound_sentence("i like cats and dogs"),
                         ["i like cats and dogs"])
        # no clause starter on the left -> never split
        self.assertEqual(split_compound_sentence("mac and cheese is great"),
                         ["mac and cheese is great"])
        # single-token right side -> never split
        self.assertEqual(split_compound_sentence("my favorite color is blue and red"),
                         ["my favorite color is blue and red"])

    def test_compound_split_is_opt_in(self):
        profile = ["i like cats and i like dogs ."]
        default_segments = segment_profile(profile, role="self")
        self.assertEqual(len(default_segments), 1)
        split_segments = segment_profile(profile, role="self", split_compound=True)
        self.assertEqual([s.text for s in split_segments], ["i like cats", "i like dogs ."])
        self.assertEqual([s.order for s in split_segments], [0, 1])

    def test_segment_record_covers_both_roles(self):
        rec = _record(
            self_profile=["i like tea ."],
            partner_profile=["i love coffee .", "i work at a bank ."],
        )
        parts = segment_record(rec)
        self.assertEqual(len(parts["self"]), 1)
        self.assertEqual(len(parts["partner"]), 2)
        self.assertEqual(parts["partner"][1].text, "i work at a bank .")

    def test_attribute_segment_round_trip_and_validation(self):
        seg = AttributeSegment(
            attribute_id="i_like_dogs_ab12cd34", text="i like dogs",
            canonical="i like dogs", role="self", order=0,
        )
        self.assertEqual(AttributeSegment.from_dict(seg.to_dict()), seg)
        with self.assertRaises(KeyError):
            AttributeSegment.from_dict({"text": "x"})
        with self.assertRaises(ValueError):
            AttributeSegment(attribute_id="NOT safe!", text="x",
                             canonical="x", role="self", order=0)
        with self.assertRaises(ValueError):
            AttributeSegment(attribute_id="ok", text="x",
                             canonical="x", role="other", order=0)


class InventoryUnitTest(unittest.TestCase):

    def _records(self):
        return [
            _record(self_profile=["i like dogs .", "i run daily ."],
                    partner_profile=["i love coffee ."], idx=0),
            _record(self_profile=["i like dogs ."],
                    partner_profile=["i read books ."], idx=1),
            _record(self_profile=["i like cats ."],
                    partner_profile=["i love coffee .", "i drink tea ."], idx=2),
        ]

    def test_counts_roles_and_distinct_profiles(self):
        inv = build_attribute_inventory(self._records(), dataset_version="original")
        by_canonical = {s.canonical: s for s in inv.attributes}

        dogs = by_canonical["i like dogs"]
        self.assertEqual(dogs.count, 2)          # record-weighted
        self.assertEqual(dogs.n_profiles, 2)     # two distinct self-profiles hold it
        self.assertEqual(dogs.n_self, 2)
        self.assertEqual(dogs.n_partner, 0)
        self.assertEqual(dogs.examples, ["i like dogs ."])

        coffee = by_canonical["i love coffee"]
        self.assertEqual(coffee.count, 2)
        self.assertEqual(coffee.n_profiles, 2)   # two distinct profiles hold it
        self.assertEqual(coffee.n_partner, 2)

        self.assertEqual(inv.dataset_version, "original")
        self.assertEqual(inv.totals["n_records"], 3)
        # unique canonicals: dogs, run daily, coffee, read books, cats, drink tea
        self.assertEqual(inv.totals["n_unique"], 6)
        self.assertEqual(inv.totals["n_kept"], 6)

    def test_sorted_by_count_then_canonical(self):
        inv = build_attribute_inventory(self._records(), dataset_version="original")
        counts = [s.count for s in inv.attributes]
        self.assertEqual(counts, sorted(counts, reverse=True))
        top = [s for s in inv.attributes if s.count == counts[0]]
        self.assertEqual([s.canonical for s in top], sorted(s.canonical for s in top))

    def test_min_count_filters_but_totals_keep_unique(self):
        inv = build_attribute_inventory(
            self._records(), dataset_version="original", min_count=2
        )
        self.assertTrue(all(s.count >= 2 for s in inv.attributes))
        self.assertEqual(inv.totals["n_unique"], 6)
        self.assertEqual(inv.totals["n_kept"], len(inv.attributes))

    def test_max_examples_caps_surface_forms(self):
        # surface variants that share ONE canonical key
        variants = ["i like dogs .", "i like dogs !", "i like dogs", "i like  dogs ."]
        records = [
            _record(self_profile=[text], idx=i)
            for i, text in enumerate(variants)
        ]
        inv = build_attribute_inventory(records, dataset_version="original",
                                        max_examples=2)
        dogs = next(s for s in inv.attributes if s.canonical == "i like dogs")
        self.assertEqual(dogs.count, 4)
        self.assertEqual(len(dogs.examples), 2)

    def test_enforced_version_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_attribute_inventory(self._records(), dataset_version="revised")

    def test_infer_version_and_mixed(self):
        same = build_attribute_inventory(self._records())
        self.assertEqual(same.dataset_version, "original")
        mixed = build_attribute_inventory(
            [self._records()[0], _record(version="revised")]
        )
        self.assertEqual(mixed.dataset_version, "mixed")

    def test_save_load_round_trip(self):
        inv = build_attribute_inventory(self._records(), dataset_version="original",
                                        params={"splits": ["train"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = save_inventory(inv, Path(tmp) / "inventory_original.json")
            loaded = load_inventory(path)
        self.assertEqual(loaded.to_dict(), inv.to_dict())

    def test_stats_validation(self):
        with self.assertRaises(ValueError):
            AttributeStats(attribute_id="a", canonical="a", count=-1,
                           n_profiles=0, n_self=0, n_partner=0)
        with self.assertRaises(KeyError):
            AttributeStats.from_dict({"attribute_id": "a"})


@unittest.skipUnless(
    (PROCESSED / "original" / "train.jsonl").exists(),
    "real processed data not built (run scripts/01_build_dataset.sh)",
)
class RealDataInventoryTest(unittest.TestCase):

    def _load(self, version):
        path = PROCESSED / version / "train.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            return [ProcessedRecord.from_dict(json.loads(line)) for line in f]

    def test_original_and_revised_inventories_from_real_data(self):
        for version in ("original", "revised"):
            records = self._load(version)
            self.assertEqual(len(records), EXPECTED_TRAIN_RECORDS)

            inv = build_attribute_inventory(records, dataset_version=version)
            self.assertEqual(inv.totals["n_records"], EXPECTED_TRAIN_RECORDS)
            self.assertEqual(
                inv.totals["n_segments"], EXPECTED_INVENTORY[version]["n_segments"]
            )
            self.assertEqual(
                inv.totals["n_unique"], EXPECTED_INVENTORY[version]["n_unique"]
            )

            ids = set()
            for stat in inv.attributes:
                self.assertRegex(stat.attribute_id, r"^[a-z0-9_]+$")
                self.assertNotIn(stat.attribute_id, ids)
                ids.add(stat.attribute_id)
                self.assertGreaterEqual(stat.count, 1)
                self.assertGreaterEqual(stat.n_profiles, 1)
                self.assertGreater(stat.n_self + stat.n_partner, 0)
                self.assertTrue(stat.examples)

            # both speakers' attributes are represented in the inventory
            self.assertTrue(any(s.n_self > 0 for s in inv.attributes))
            self.assertTrue(any(s.n_partner > 0 for s in inv.attributes))


if __name__ == "__main__":
    unittest.main()
