"""Tests for Component B composition and conflict experiments."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from duet.steering.apply import Direction
from duet.steering.composition import composition_points, composition_sets, run_composition_experiment


def _direction(attr, layer):
    return Direction(attr, attr, layer, np.array([1, 0], dtype=np.float32), 256,
                     "Qwen/Qwen2.5-0.5B-Instruct", "original", "self", "last_token")


class CompositionTest(unittest.TestCase):
    def test_sets_include_capped_composition_and_conflict(self):
        sets = composition_sets(["married", "single", "vegan"], sizes=[2, 3],
                                conflicting_pair=["married", "single"], max_sets_per_size=1)
        self.assertEqual(len(sets), 3)
        self.assertIn((("married", "single"), "conflict"), sets)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_composition_runs_and_persists_distinct_results(self):
        import torch

        class Block(torch.nn.Module):
            def forward(self, hidden): return hidden
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([Block()])
        points = composition_points(layers=[0], magnitudes=[1.0],
                                    sets=[(("married", "single"), "conflict")])
        directions = {("married", 0): _direction("married", 0), ("single", 0): _direction("single", 0)}
        with tempfile.TemporaryDirectory() as tmp:
            records = run_composition_experiment(Model(), directions=directions, points=points,
                generate=lambda _model, _point: ["x"], output_dir=Path(tmp) / "composition")
            self.assertEqual(records[0]["condition"], "conflict")
            saved = json.loads((Path(tmp) / "composition" / "composition_results.json").read_text())
            self.assertEqual(saved[0]["n_attributes"], 2)


if __name__ == "__main__":
    unittest.main()
