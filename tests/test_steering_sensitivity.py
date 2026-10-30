"""Tests for the true-vs-perturbed profile log-probability definition."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from duet.steering.sensitivity import SensitivityCase, conditional_target_logprob, persona_sensitivity_delta, run_sensitivity_experiment


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
class SensitivityTest(unittest.TestCase):
    def setUp(self):
        import torch

        class Tokenizer:
            table = {"true": 1, "perturbed": 2, "answer": 3}
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [self.table[token] for token in text.split()]}

        class Model(torch.nn.Module):
            def forward(self, input_ids):
                logits = torch.zeros((*input_ids.shape, 4))
                # A target following "true" is more likely than one following "perturbed".
                for index in range(input_ids.shape[1]):
                    logits[:, index, 3] = torch.where(input_ids[:, index] == 1, 4.0, -4.0)
                return type("Output", (), {"logits": logits})()

        self.tokenizer, self.model = Tokenizer(), Model()

    def test_true_profile_has_positive_delta(self):
        record = persona_sensitivity_delta(self.model, self.tokenizer,
            SensitivityCase("x", "true", "perturbed", "answer"))
        self.assertGreater(record["logprob_delta"], 0)
        self.assertEqual(record["true_target_tokens"], 1)

    def test_invalid_target_boundary_rejected(self):
        import torch
        with self.assertRaises(ValueError):
            conditional_target_logprob(self.model, torch.tensor([[1]]), target_start=1)

    def test_results_are_written_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_sensitivity_experiment(self.model, self.tokenizer,
                [SensitivityCase("x", "true", "perturbed", "answer")], output_dir=Path(tmp) / "sensitivity")
            self.assertEqual(len(rows), 1)
            with (Path(tmp) / "sensitivity" / "sensitivity_results.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
