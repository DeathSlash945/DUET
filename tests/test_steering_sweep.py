"""Tests for Component B layer × magnitude sweep orchestration."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from duet.steering.apply import Direction
from duet.steering.sweep import run_sweep, sweep_points, write_sweep_results


def _direction(attr, layer, vector):
    return Direction(attr, attr, layer, np.asarray(vector, dtype=np.float32), 256,
                     "Qwen/Qwen2.5-0.5B-Instruct", "original", "self", "last_token")


class SweepTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_hooks_run_all_layer_magnitude_points_and_write_results(self):
        import torch

        class Block(torch.nn.Module):
            def forward(self, hidden):
                return hidden

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Block(), Block()])

            def forward(self, hidden):
                for block in self.model.layers:
                    hidden = block(hidden)
                return hidden

        model = Model()
        directions = {
            ("a", 0): _direction("a", 0, [1, 0]), ("a", 1): _direction("a", 1, [0, 1]),
        }
        points = sweep_points(layers=[0, 1], magnitudes=[0.5, 1.0], attribute_ids=["a"])

        def generate(active_model, _point):
            return [active_model(torch.zeros((1, 2, 2))).detach().numpy().tolist()]

        with tempfile.TemporaryDirectory() as tmp:
            records = run_sweep(model, directions=directions, points=points, generate=generate,
                                score=lambda _items, point: {"abs_magnitude": abs(point.magnitude)},
                                output_dir=Path(tmp) / "results")
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["n_generated"], 1)
            data = json.loads((Path(tmp) / "results" / "sweep_results.json").read_text())
            self.assertEqual(len(data), 4)
            with (Path(tmp) / "results" / "sweep_results.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_result_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            write_sweep_results([{"layer": 0, "magnitude": 1.0}], output)
            with self.assertRaises(FileExistsError):
                write_sweep_results([{"layer": 0, "magnitude": 1.0}], output)


if __name__ == "__main__":
    unittest.main()
