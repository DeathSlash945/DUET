"""Tests for Component B inference-time steering hooks."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from duet.steering.apply import (
    Direction,
    compose_directions,
    discover_directions,
    install_steering_hooks,
    read_direction,
)
from duet.steering.extract import save_direction


def _direction(attribute_id, layer, vector):
    return Direction(attribute_id, attribute_id, layer, np.asarray(vector, dtype=np.float32), 256,
                     "Qwen/Qwen2.5-0.5B-Instruct", "original", "self", "last_token")


class DirectionTest(unittest.TestCase):
    def test_composition_normalizes_then_sums(self):
        vectors = compose_directions(
            [_direction("a", 0, [3, 4]), _direction("b", 0, [0, 2])],
            magnitudes={"a": 2.0, "b": -1.0},
        )
        np.testing.assert_allclose(vectors[0], [1.2, 0.6])

    def test_zero_direction_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            compose_directions([_direction("a", 0, [0, 0])], magnitudes=1.0)

    def test_discovery_requires_exactly_one_file_per_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for layer in (0, 23):
                save_direction(root / str(layer) / "a.npz", direction=np.ones(4), layer=layer,
                               attribute_id="a", canonical="a", n_pairs=256,
                               model_id="Qwen/Qwen2.5-0.5B-Instruct", dataset_version="original")
            found = discover_directions(root, attribute_ids=["a"], layers=[0, 23],
                                        expected_model="Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(set(found), {("a", 0), ("a", 23)})
            self.assertEqual(read_direction(root / "0" / "a.npz").vector.shape, (4,))


class HookTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_last_token_hook_and_cleanup(self):
        import torch

        class Block(torch.nn.Module):
            def forward(self, hidden):
                return hidden + 1

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block(), Block()])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Core()

            def forward(self, hidden):
                for block in self.model.layers:
                    hidden = block(hidden)
                return hidden

        model = Model()
        source = torch.zeros((1, 3, 2))
        baseline = model(source)
        with install_steering_hooks(model, {0: np.array([2, -1], dtype=np.float32)}):
            steered = model(source)
        restored = model(source)
        self.assertTrue(torch.equal(baseline, restored))
        self.assertTrue(torch.equal(steered[:, :2], baseline[:, :2]))
        self.assertTrue(torch.equal(steered[:, -1], baseline[:, -1] + torch.tensor([2.0, -1.0])))

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_tuple_block_output_is_preserved(self):
        import torch

        class TupleBlock(torch.nn.Module):
            def forward(self, hidden):
                return hidden, "cache"

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([TupleBlock()])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Core()

        model = Model()
        with install_steering_hooks(model, {0: np.array([1, 1], dtype=np.float32)}):
            hidden, cache = model.model.layers[0](torch.zeros((1, 2, 2)))
        self.assertEqual(cache, "cache")
        self.assertTrue(np.allclose(hidden.detach().numpy()[0, -1], [1, 1]))


if __name__ == "__main__":
    unittest.main()
