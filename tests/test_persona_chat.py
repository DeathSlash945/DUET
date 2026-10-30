"""
Unit and integration tests for src/duet/data/persona_chat.py
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from duet.data.persona_chat import (
    parse_parlai_episode,
    parse_raw_personachat_sample,
    process_dataset_split,
)
from duet.data.schemas import ProcessedRecord


class TestPersonaChatPipeline(unittest.TestCase):

    def test_parse_parlai_episode_both_profiles(self):
        # Synthetic ParlAI episode in both_original / both_revised format
        parlai_episode = [
            {
                "text": (
                    "your persona: i have two dogs .\n"
                    "your persona: i like hiking .\n"
                    "partner's persona: i love playing guitar .\n"
                    "partner's persona: i live in NYC .\n"
                    "hi ! how are you ?"
                ),
                "labels": ["i am great , just walked my dogs !"]
            },
            {
                "text": "that sounds fun !",
                "labels": ["yeah , do you play any instruments ?"]
            }
        ]

        records = parse_parlai_episode(
            episode_turns=parlai_episode,
            episode_index=0,
            dataset_version="original",
            split="train"
        )

        self.assertEqual(len(records), 2)
        
        rec1 = records[0]
        self.assertEqual(rec1.profile_self, ["i have two dogs .", "i like hiking ."])
        self.assertEqual(rec1.profile_partner, ["i love playing guitar .", "i live in NYC ."])
        self.assertEqual(rec1.history, ["hi ! how are you ?"])
        self.assertEqual(rec1.target, "i am great , just walked my dogs !")
        self.assertEqual(rec1.dataset_version, "original")

        rec2 = records[1]
        self.assertEqual(rec2.profile_self, ["i have two dogs .", "i like hiking ."])
        self.assertEqual(rec2.profile_partner, ["i love playing guitar .", "i live in NYC ."])
        self.assertEqual(rec2.history, ["hi ! how are you ?", "i am great , just walked my dogs !", "that sounds fun !"])
        self.assertEqual(rec2.target, "yeah , do you play any instruments ?")

    def test_reject_missing_profiles(self):
        # Episode with missing partner persona
        malformed_episode = [
            {
                "text": "your persona: i like cats .\nhello !",
                "labels": ["hi !"]
            }
        ]

        with self.assertRaises(ValueError) as ctx:
            parse_parlai_episode(
                episode_turns=malformed_episode,
                episode_index=1,
                dataset_version="original",
                split="train"
            )
        self.assertIn("Missing speaker profile", str(ctx.exception))

    def test_dict_format_backward_compatibility(self):
        dict_sample = {
            "personality": ["i like tea ."],
            "partner_persona": ["i love coffee ."],
            "history": ["hello !"],
            "target": "hi !",
            "persona_labels": [0]
        }

        records = parse_raw_personachat_sample(
            raw_sample=dict_sample,
            sample_index=10,
            dataset_version="revised",
            split="valid"
        )

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.profile_self, ["i like tea ."])
        self.assertEqual(rec.profile_partner, ["i love coffee ."])
        self.assertEqual(rec.dataset_version, "revised")

    def test_process_dataset_split_to_jsonl(self):
        parlai_episode = [
            {
                "text": "your persona: i like coding .\npartner's persona: i like painting .\nhi !",
                "labels": ["hello !"]
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = process_dataset_split(
                dataset_split=[parlai_episode],
                dataset_version="original",
                split_name="train",
                output_dir=tmpdir
            )

            self.assertTrue(Path(out_file).exists())

            with open(out_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                data = json.loads(lines[0])
                rec = ProcessedRecord.from_dict(data)
                self.assertEqual(rec.profile_self, ["i like coding ."])
                self.assertEqual(rec.profile_partner, ["i like painting ."])


if __name__ == "__main__":
    unittest.main()
