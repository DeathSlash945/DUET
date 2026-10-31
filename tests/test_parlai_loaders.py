"""
Tests for the official ParlAI fbdialog loaders (both_original / both_revised).

Fixtures mirror the byte-level format of the official personachat release:
one numbered line per sentence/turn, tab-separated label/reward/candidate
columns, and an episode restart at conv id 1. No network access is required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from duet.data.persona_chat import (
    BOTH_FILENAMES,
    parse_raw_personachat_sample,
    process_parlai_personachat,
    read_parlai_txt_file,
)
from duet.data.schemas import ProcessedRecord

T = "\t"

# Two episodes: the second contains an unlabelled context line (no tab).
ORIGINAL_FIXTURE = (
    "1 your persona: i read twenty books a year.\n"
    "2 your persona: i only eat kosher.\n"
    "3 partner's persona: horror movies are my favorites.\n"
    "4 partner's persona: i have a son who is in junior high school.\n"
    f"5 hello what are doing today ?{T}i am good , i just got off work .{T}{T}"
    "irrelevant distractor|i am good , i just got off work .\n"
    f"6 i just got done watching a horror movie{T}i rather read , i have read "
    f"about 20 books this year .{T}{T}distractor|i rather read , i have read "
    "about 20 books this year .\n"
    "1 your persona: i love to remodel homes.\n"
    "2 partner's persona: i race large felines who are in captivity to remain healthy.\n"
    f"3 hi , how are you doing ?{T}i am getting ready to do some cheetah chasing "
    "to stay in shape .\n"
    "4 what do you do for fun ?\n"
    f"5 i enjoy romantic movie .{T}that is cool , i prefer comedies myself .\n"
)

# Revised version of episode 1: rephrased personas, as in the official release.
REVISED_FIXTURE = (
    "1 your persona: i read twenty tomes yearly.\n"
    "2 partner's persona: slasher cinema ranks first for me.\n"
    f"3 hello what are doing today ?{T}i am fine , just finished work .{T}{T}"
    "i am fine , just finished work .\n"
)

EXPECTED_SELF_0 = ["i read twenty books a year.", "i only eat kosher."]
EXPECTED_PARTNER_0 = [
    "horror movies are my favorites.",
    "i have a son who is in junior high school.",
]


class FbDialogReaderTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def _write(self, name, content):
        path = self.tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_splits_episodes_on_conv_id_reset(self):
        path = self._write("both.txt", ORIGINAL_FIXTURE)
        episodes = read_parlai_txt_file(path)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(len(episodes[0]), 6)
        self.assertEqual(len(episodes[1]), 5)

        first = episodes[0][0]
        self.assertEqual(first["text"], "your persona: i read twenty books a year.")
        self.assertEqual(first["labels"], [])  # context line: no record downstream

        labelled = episodes[0][4]
        self.assertEqual(labelled["labels"], ["i am good , i just got off work ."])
        # candidate column must not leak into text or labels
        self.assertNotIn("irrelevant distractor", labelled["text"])
        self.assertNotIn("irrelevant distractor", labelled["labels"][0])

    def test_records_preserve_both_profiles_and_history(self):
        path = self._write("both.txt", ORIGINAL_FIXTURE)
        episodes = read_parlai_txt_file(path)

        records = parse_raw_personachat_sample(episodes[0], 0, "original", "train")
        self.assertEqual(len(records), 2)

        rec1 = records[0]
        self.assertEqual(rec1.profile_self, EXPECTED_SELF_0)
        self.assertEqual(rec1.profile_partner, EXPECTED_PARTNER_0)
        self.assertEqual(rec1.history, ["hello what are doing today ?"])
        self.assertEqual(rec1.target, "i am good , i just got off work .")
        self.assertEqual(rec1.dataset_version, "original")
        self.assertEqual(rec1.persona_labels, [])  # no persona indices upstream

        rec2 = records[1]
        # both profiles present on EVERY record (Component C invariant)
        self.assertEqual(rec2.profile_self, EXPECTED_SELF_0)
        self.assertEqual(rec2.profile_partner, EXPECTED_PARTNER_0)
        self.assertEqual(
            rec2.history,
            [
                "hello what are doing today ?",
                "i am good , i just got off work .",
                "i just got done watching a horror movie",
            ],
        )
        self.assertEqual(
            rec2.target, "i rather read , i have read about 20 books this year ."
        )
        self.assertNotEqual(rec1.id, rec2.id)

    def test_unlabelled_context_line_kept_in_history_without_record(self):
        path = self._write("both.txt", ORIGINAL_FIXTURE)
        episodes = read_parlai_txt_file(path)

        records = parse_raw_personachat_sample(episodes[1], 1, "original", "train")
        # 5 turns but only 2 labelled ones become records
        self.assertEqual(len(episodes[1]), 5)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[1].history,
            [
                "hi , how are you doing ?",
                "i am getting ready to do some cheetah chasing to stay in shape .",
                "what do you do for fun ?",
                "i enjoy romantic movie .",
            ],
        )
        self.assertEqual(
            records[1].target, "that is cool , i prefer comedies myself ."
        )

    def test_literal_newline_unescaped_like_parlai(self):
        content = (
            "1 your persona: i like to ski.\n"
            "2 partner's persona: i race cheetahs at the zoo.\n"
            f"3 i love the mountain\\nand the snow{T}me too !{T}{T}me too !\n"
        )
        path = self._write("both.txt", content)
        episodes = read_parlai_txt_file(path)

        records = parse_raw_personachat_sample(episodes[0], 0, "original", "train")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].history, ["i love the mountain\nand the snow"])
        self.assertEqual(records[0].target, "me too !")

    def test_missing_partner_profile_raises(self):
        content = (
            "1 your persona: i like cats .\n"
            f"2 hello !{T}hi !\n"
        )
        path = self._write("both.txt", content)
        episodes = read_parlai_txt_file(path)

        with self.assertRaises(ValueError) as ctx:
            parse_raw_personachat_sample(episodes[0], 0, "original", "train")
        self.assertIn("Missing speaker profile", str(ctx.exception))

    def test_malformed_lines_skipped_not_fatal(self):
        content = (
            "garbage without a conv id\n"
            "1 your persona: i like cats .\n"
            "2 partner's persona: i like dogs .\n"
            f"3 hello !{T}hi !\n"
        )
        path = self._write("both.txt", content)
        episodes = read_parlai_txt_file(path)

        self.assertEqual(len(episodes), 1)
        records = parse_raw_personachat_sample(episodes[0], 0, "original", "train")
        self.assertEqual(len(records), 1)


class ProcessParlaiPersonachatTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.raw_dir = base / "raw"
        self.output_dir = base / "processed"
        self.personachat_dir = self.raw_dir / "personachat"
        self.personachat_dir.mkdir(parents=True)

    def _install_raw_files(self, dataset_version, fixture, splits):
        for split in splits:
            path = self.personachat_dir / f"{split}_both_{dataset_version}.txt"
            path.write_text(fixture, encoding="utf-8")

    def _read_jsonl(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [ProcessedRecord.from_dict(json.loads(line)) for line in f]

    def test_original_and_revised_end_to_end(self):
        self._install_raw_files("original", ORIGINAL_FIXTURE, ("train", "valid", "test"))
        # revised test split intentionally absent: must be skipped, not fatal
        self._install_raw_files("revised", REVISED_FIXTURE, ("train", "valid"))

        original_paths = process_parlai_personachat(
            dataset_version="original", raw_dir=self.raw_dir, output_dir=self.output_dir
        )
        revised_paths = process_parlai_personachat(
            dataset_version="revised", raw_dir=self.raw_dir, output_dir=self.output_dir
        )

        self.assertEqual(len(original_paths), 3)
        self.assertEqual(len(revised_paths), 2)

        # Original: 2 episodes x 2 labelled turns = 4 records per split
        train_records = self._read_jsonl(self.output_dir / "original" / "train.jsonl")
        self.assertEqual(len(train_records), 4)
        for rec in train_records:
            self.assertTrue(rec.profile_self)
            self.assertTrue(rec.profile_partner)
            self.assertEqual(rec.dataset_version, "original")
        self.assertEqual(train_records[0].profile_self, EXPECTED_SELF_0)
        self.assertEqual(train_records[0].profile_partner, EXPECTED_PARTNER_0)

        # Revised: 1 episode x 1 labelled turn, marked and distinguishable
        revised_records = self._read_jsonl(self.output_dir / "revised" / "train.jsonl")
        self.assertEqual(len(revised_records), 1)
        rec = revised_records[0]
        self.assertEqual(rec.dataset_version, "revised")
        self.assertEqual(rec.profile_self, ["i read twenty tomes yearly."])
        self.assertEqual(
            rec.profile_partner, ["slasher cinema ranks first for me."]
        )
        self.assertNotEqual(rec.profile_self, train_records[0].profile_self)

        # Both versions coexist side by side; absent split produced nothing
        self.assertTrue((self.output_dir / "original" / "test.jsonl").exists())
        self.assertFalse((self.output_dir / "revised" / "test.jsonl").exists())

    def test_missing_split_files_raise_file_not_found(self):
        # personachat dir exists (so no download is attempted) but is empty
        with self.assertRaises(FileNotFoundError) as ctx:
            process_parlai_personachat(
                dataset_version="original",
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
            )
        self.assertIn("train_both_original.txt", str(ctx.exception))

    def test_expected_filenames_match_official_layout(self):
        self.assertEqual(
            BOTH_FILENAMES["original"],
            ("train_both_original.txt", "valid_both_original.txt", "test_both_original.txt"),
        )
        self.assertEqual(
            BOTH_FILENAMES["revised"],
            ("train_both_revised.txt", "valid_both_revised.txt", "test_both_revised.txt"),
        )


if __name__ == "__main__":
    unittest.main()
