"""
Tests for src/duet/steering/extract.py — Component B paired-activation collection.

Logic tests (config, pairing, selection, mean_diff, npz contract) run without
torch. The end-to-end smoke test uses a tiny random local model and is skipped
until torch/transformers/tokenizers are installed (approval-gated) and real
processed data exists.
"""

import json
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from importlib.util import find_spec
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from duet.data.attributes import AttributeInventory, AttributeStats, load_inventory
from duet.data.schemas import ProcessedRecord
from duet.steering.extract import (
    _load_policy,
    build_pairs,
    extract_settings,
    load_direction,
    load_steer_config,
    mean_diff,
    profile_attribute_set,
    render_prompt,
    run_smoke,
    save_direction,
    select_attributes,
)
from duet.training.prompting import format_prompt

ROOT = Path(__file__).resolve().parents[1]
REAL_RECORDS = ROOT / "data" / "processed" / "original" / "train.jsonl"
REAL_INVENTORY = ROOT / "data" / "attrs" / "inventory_original.json"
REAL_DATA = REAL_RECORDS.exists() and REAL_INVENTORY.exists()
ML_STACK = all(find_spec(m) for m in ("torch", "transformers", "tokenizers"))


def _rec(idx, self_profile, partner_profile=None, history=None):
    return ProcessedRecord(
        id=f"original_train_ep{idx:05d}_t00",
        history=history if history is not None else ["hello !"],
        profile_self=self_profile,
        profile_partner=partner_profile if partner_profile is not None else ["i read books ."],
        target="hi !",
        persona_labels=[],
        dataset_version="original",
    )


class ConfigTest(unittest.TestCase):

    def test_real_steer_config_extracts_expected_settings(self):
        cfg = extract_settings(load_steer_config(ROOT / "configs" / "steer.yaml"))
        self.assertEqual(cfg["normalization"], "mean_diff")
        self.assertEqual(cfg["profile_side"], "self")
        self.assertEqual(cfg["activation"], "last_token")
        self.assertEqual(cfg["pairs_per_attribute"], 256)
        self.assertEqual(cfg["context_length"], 256)
        self.assertEqual(cfg["selection_top_k"], 0)
        self.assertEqual(cfg["batch_size"], 8)
        self.assertEqual(cfg["activations_path"], "data/attrs")
        steer = load_steer_config(ROOT / "configs" / "steer.yaml")
        self.assertTrue(steer.get("base_model"))

    def test_defaults_apply_when_section_missing(self):
        cfg = extract_settings({})
        self.assertEqual(cfg["pairs_per_attribute"], 256)
        self.assertEqual(cfg["profile_side"], "self")

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            extract_settings({"extract": {"normalization": "l2"}})
        with self.assertRaises(ValueError):
            extract_settings({"extract": {"profile_side": "teacher"}})
        with self.assertRaises(ValueError):
            extract_settings({"extract": {"activation": "mean_pool"}})
        with self.assertRaises(ValueError):
            extract_settings({"extract": {"pairs_per_attribute": 0}})
        with self.assertRaises(ValueError):
            extract_settings({"extract": {"selection_top_k": -1}})
        with self.assertRaises(KeyError):
            extract_settings({"extract": {"bogus_key": 1}})


class PolicyLoadTest(unittest.TestCase):

    def test_cuda_requires_cuda_enabled_torch_before_model_download(self):
        # The real environment may be CPU-only.  CUDA must fail before trying
        # to resolve/download a model, rather than silently using the CPU.
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cannot access CUDA"):
                _load_policy("not-a-real-model", "cuda")


class PromptTest(unittest.TestCase):

    def test_render_prompt_is_the_training_prompt(self):
        rec = _rec(0, ["i like tea ."], ["i love coffee ."], history=["hello !"])
        self.assertEqual(
            render_prompt(rec),
            format_prompt(rec.history, rec.profile_self, rec.profile_partner),
        )
        self.assertTrue(render_prompt(rec).endswith("assistant:"))


class PairingTest(unittest.TestCase):

    def _records(self):
        return [
            _rec(1, ["i like dogs ."], history=["a", "b"]),
            _rec(2, ["i like cats ."], history=["x", "y"]),
            _rec(3, ["i love coffee ."], partner_profile=["i like dogs ."],
                 history=["p", "q"]),
            _rec(4, ["i like dogs .", "i run ."], history=["1", "2", "3", "4", "5"]),
            _rec(5, ["i like dogs ."], history=[]),          # unusable: no history
        ]

    def test_expressed_matched_and_contamination_guard(self):
        result = build_pairs(self._records(), "i like dogs",
                             profile_side="self", max_pairs=10)
        # r5 skipped (empty history); r3 excluded (attribute on other side only)
        self.assertEqual(result.n_expressed, 2)   # r1, r4
        self.assertEqual(result.n_matched, 1)     # r2
        self.assertEqual(len(result.pairs), 1)
        expressed, matched = result.pairs[0]
        self.assertEqual(expressed.id, _rec(1, ["x"]).id)
        self.assertEqual(matched.id, _rec(2, ["x"]).id)

    def test_either_side_admits_partner_only_records(self):
        result = build_pairs(self._records(), "i like dogs",
                             profile_side="either", max_pairs=10)
        self.assertEqual(result.n_expressed, 3)   # r1, r3, r4

    def test_deterministic_and_max_pairs_respected(self):
        records = self._records() + [_rec(6, ["i walk ."], history=["m", "n"])]
        first = build_pairs(records, "i like dogs", profile_side="self", max_pairs=10)
        second = build_pairs(records, "i like dogs", profile_side="self", max_pairs=10)
        self.assertEqual(
            [(e.id, m.id) for e, m in first.pairs],
            [(e.id, m.id) for e, m in second.pairs],
        )
        limited = build_pairs(records, "i like dogs", profile_side="self", max_pairs=1)
        self.assertEqual(len(limited.pairs), 1)

    def test_absent_attribute_yields_empty_pairing(self):
        result = build_pairs(self._records(), "i play chess")
        self.assertEqual(result.pairs, [])       # no expressed candidates
        self.assertEqual(result.n_expressed, 0)
        self.assertEqual(result.n_matched, 4)    # all usable records lack it (r5 skipped)

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            build_pairs(self._records(), "i like dogs", profile_side="wrong")
        with self.assertRaises(ValueError):
            build_pairs(self._records(), "i like dogs", max_pairs=0)
        with self.assertRaises(ValueError):
            build_pairs(self._records(), "")

    def test_matched_profiles_lack_attribute_on_both_sides(self):
        result = build_pairs(self._records(), "i like dogs",
                             profile_side="self", max_pairs=10)
        for _, matched in result.pairs:
            self.assertNotIn("i like dogs",
                             profile_attribute_set(matched.profile_self))
            self.assertNotIn("i like dogs",
                             profile_attribute_set(matched.profile_partner))


class SelectionTest(unittest.TestCase):

    def _inventory(self):
        stats = [
            AttributeStats(attribute_id="b_x", canonical="b", count=510,
                           n_profiles=5, n_self=10, n_partner=500),
            AttributeStats(attribute_id="a_x", canonical="a", count=300,
                           n_profiles=5, n_self=300, n_partner=0),
            AttributeStats(attribute_id="c_x", canonical="c", count=150,
                           n_profiles=5, n_self=150, n_partner=0),
        ]
        return AttributeInventory(dataset_version="original", params={},
                                  totals={}, attributes=stats)

    def test_side_specific_filtering_and_sort(self):
        inv = self._inventory()
        self.assertEqual(
            [s.canonical for s in select_attributes(
                inv, profile_side="self", pairs_per_attribute=100)],
            ["a", "c"],
        )
        self.assertEqual(
            [s.canonical for s in select_attributes(
                inv, profile_side="partner", pairs_per_attribute=100)],
            ["b"],
        )
        self.assertEqual(
            [s.canonical for s in select_attributes(
                inv, profile_side="either", pairs_per_attribute=100)],
            ["b", "a", "c"],   # sorted by count desc despite shuffled input
        )

    def test_top_k_and_validation(self):
        inv = self._inventory()
        self.assertEqual(
            [s.canonical for s in select_attributes(
                inv, profile_side="self", pairs_per_attribute=1, top_k=1)],
            ["a"],
        )
        with self.assertRaises(ValueError):
            select_attributes(inv, profile_side="nope")


class MeanDiffTest(unittest.TestCase):

    def test_mean_diff_values_and_dtype(self):
        expressed = {
            0: np.array([[1.0, 2.0], [3.0, 4.0]]),
            1: np.array([[0.0, 0.0], [2.0, 2.0]]),
        }
        matched = {
            0: np.array([[0.0, 1.0], [1.0, 1.0]]),
            1: np.array([[1.0, 1.0], [1.0, 1.0]]),
        }
        directions = mean_diff(expressed, matched)
        np.testing.assert_allclose(directions[0], np.array([1.5, 2.0]))
        np.testing.assert_allclose(directions[1], np.array([0.0, 0.0]))
        self.assertEqual(directions[0].dtype, np.float32)

    def test_mean_diff_errors(self):
        with self.assertRaises(ValueError):
            mean_diff({}, {})
        with self.assertRaises(ValueError):
            mean_diff({0: np.zeros((2, 4))}, {1: np.zeros((2, 4))})
        with self.assertRaises(ValueError):
            mean_diff({0: np.zeros((2, 4))}, {0: np.zeros((3, 4))})
        with self.assertRaises(ValueError):
            mean_diff({0: np.zeros((0, 4))}, {0: np.zeros((0, 4))})


class DirectionNpzTest(unittest.TestCase):

    def test_round_trip_and_path_structure(self):
        direction = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_direction(
                Path(tmp) / "6" / "i_like_dogs_ab12cd34.npz",
                direction=direction, layer=6,
                attribute_id="i_like_dogs_ab12cd34", canonical="i like dogs",
                n_pairs=256, model_id="test-model", profile_side="self",
                dataset_version="original",
            )
            path = Path(path)
            self.assertEqual(path.parent.name, "6")             # {layer}/ contract
            self.assertEqual(path.name, "i_like_dogs_ab12cd34.npz")
            loaded = load_direction(path)

        np.testing.assert_allclose(loaded["direction"], direction)
        self.assertEqual(int(loaded["layer"].item()), 6)
        self.assertEqual(loaded["attribute_id"].item(), "i_like_dogs_ab12cd34")
        self.assertEqual(loaded["canonical"].item(), "i like dogs")
        self.assertEqual(int(loaded["n_pairs"].item()), 256)
        self.assertEqual(loaded["normalization"].item(), "mean_diff")
        self.assertEqual(loaded["model_id"].item(), "test-model")
        self.assertEqual(loaded["profile_side"].item(), "self")
        self.assertEqual(loaded["dataset_version"].item(), "original")
        self.assertEqual(loaded["activation"].item(), "last_token")

    def test_attribute_ids_are_filename_safe(self):
        for canonical in ("i like dogs", "my favorite color is blue", "i am 5 feet"):
            aid = re.sub(r"[^a-z0-9]+", "_", canonical).strip("_")
            self.assertRegex(aid, r"^[a-z0-9_]+$")


@unittest.skipUnless(REAL_DATA, "real processed data + inventory not built")
class RealDataPairingTest(unittest.TestCase):

    def test_pairs_from_real_train_split(self):
        with open(REAL_RECORDS, "r", encoding="utf-8") as f:
            records = [ProcessedRecord.from_dict(json.loads(line)) for line in f]
        inventory = load_inventory(REAL_INVENTORY)
        stat = inventory.attributes[0]

        result = build_pairs(records, stat.canonical,
                             profile_side="self", max_pairs=16)
        self.assertEqual(len(result.pairs), 16)
        for expressed, matched in result.pairs:
            self.assertIn(stat.canonical,
                          profile_attribute_set(expressed.profile_self))
            self.assertNotIn(stat.canonical,
                             profile_attribute_set(matched.profile_self))
            self.assertNotIn(stat.canonical,
                             profile_attribute_set(matched.profile_partner))
            self.assertTrue(expressed.history and matched.history)


@unittest.skipUnless(
    ML_STACK and REAL_DATA,
    "torch/transformers/tokenizers not installed (approval-gated) + real data required",
)
class SmokeExtractionTest(unittest.TestCase):

    def test_smoke_end_to_end_and_deterministic(self):
        # All assertions must run INSIDE the block: files are deleted on exit.
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "run1"
            out2 = Path(tmp) / "run2"
            attrs_root = ROOT / "data" / "attrs"
            entries_before = sorted(p.name for p in attrs_root.iterdir())

            paths1 = run_smoke(out1, seed=0, pairs=4, top_k=2)
            paths2 = run_smoke(out2, seed=0, pairs=4, top_k=2)

            # smoke must never write into the real activations contract path
            self.assertEqual(
                sorted(p.name for p in attrs_root.iterdir()), entries_before
            )

            # 2 attributes x 2 blocks of the tiny model
            self.assertEqual(len(paths1), 4)
            rel1 = [str(Path(p).relative_to(out1)) for p in paths1]
            rel2 = [str(Path(p).relative_to(out2)) for p in paths2]
            self.assertEqual(rel1, rel2)

            for rel in rel1:
                layer = int(Path(rel).parts[0])   # OS-agnostic: {layer}/{id}.npz
                self.assertIn(layer, (0, 1))
                first = load_direction(out1 / rel)
                second = load_direction(out2 / rel)
                self.assertEqual(first["direction"].shape, (32,))
                self.assertTrue(np.isfinite(first["direction"]).all())
                self.assertEqual(int(first["n_pairs"].item()), 4)
                self.assertRegex(first["attribute_id"].item(), r"^[a-z0-9_]+$")
                # same seed + same data -> identical weights -> identical directions
                np.testing.assert_allclose(first["direction"], second["direction"],
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
