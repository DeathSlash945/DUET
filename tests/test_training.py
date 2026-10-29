"""Wrapper tests: prompt contract, config mapping, PAL→contract conversion. No GPU needed."""

import json

from duet.baselines.pal.to_contract import convert_dir
from duet.training.common import config_args, read_jsonl, to_ns, write_jsonl
from duet.training.prompting import format_prompt, format_sft_text


def test_prompt_has_both_profiles_and_roles():
    p = format_prompt(["hi", "hello"], ["i hike"], ["i run"])
    assert "PERSONA (you):\n- i hike" in p
    assert "PARTNER (them):\n- i run" in p
    assert p.index("user: hi") < p.index("assistant: hello")
    assert p.rstrip().endswith("assistant:")


def test_prompt_partner_section_omitted_when_empty():
    # baseline (self-only) runs differ from bidirectional by data alone
    p = format_prompt(["hi"], ["i hike"], [])
    assert "PARTNER" not in p
    assert "PERSONA (you):" in p


def test_sft_text_ends_with_target():
    t = format_sft_text({"history": ["hi"], "profile_self": ["a"],
                         "profile_partner": [], "target": "yo"})
    assert t.endswith("yo")


def test_config_field_mapping_and_intersection():
    from trl import SFTConfig
    args = config_args({"epochs": 3, "lr": 1e-4, "per_device_batch_size": 2,
                        "grad_accum": 4, "seed": 42, "unknown_key": 1}, SFTConfig)
    assert args["num_train_epochs"] == 3
    assert args["learning_rate"] == 1e-4
    assert args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 4
    assert "unknown_key" not in args


def test_to_ns_nested_access():
    cfg = to_ns({"train": {"lr": 1e-4}, "data": {"max_seq_len": 512}})
    assert cfg.train.lr == 1e-4
    assert cfg.data.max_seq_len == 512


def test_jsonl_roundtrip(tmp_path):
    rows = [{"id": "a", "n": 1}, {"id": "b", "n": 2}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(tmp_path / "x.jsonl") == rows


def test_convert_pal_to_contract(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    pal = [["i like dogs.", "i code."], ["hi", "hello there"],
           "my favorite is python .", [1]]
    json.dump([pal], open(src / "train_origin.txt", "w"))

    stats = convert_dir(src, out)
    assert stats == {"train.jsonl": 1}
    row = read_jsonl(out / "train.jsonl")[0]
    for key in ("id", "history", "profile_self", "profile_partner",
                "target", "persona_labels", "dataset_version"):
        assert key in row, f"missing contract field: {key}"
    assert row["profile_partner"] == []       # PAL format limitation, by design
    assert row["dataset_version"] == "origin"
    assert row["history"] == ["hi", "hello there"]
