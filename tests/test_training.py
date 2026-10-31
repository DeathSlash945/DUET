"""Wrapper tests: prompt contract, config mapping, PAL→contract conversion. No GPU needed."""

import json

import pytest

from duet.baselines.pal.to_contract import convert_dir
from duet.baselines.pal.to_pairs import build_pairs as pal_build_pairs
from duet.baselines.pal.to_pairs import join_samples as pal_join_samples
from duet.training.common import (apply_profile_scope, config_args,
                                  read_jsonl, to_ns, write_jsonl)
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


# ---------------- Component C: profile scope (bidir vs self-only) ----------

def test_apply_profile_scope_keeps_both_when_true():
    recs = [{"id": "a", "profile_self": ["x"], "profile_partner": ["y"]}]
    assert apply_profile_scope(recs, True) is recs   # default path: no copy


def test_apply_profile_scope_strips_without_mutating():
    recs = [{"id": "a", "profile_self": ["x"], "profile_partner": ["y"]}]
    out = apply_profile_scope(recs, False)
    assert out[0]["profile_partner"] == []
    assert recs[0]["profile_partner"] == ["y"]       # input untouched


def test_selfonly_vs_bidir_differ_only_by_partner_section():
    """README claim 3 hygiene: conditions differ by DATA alone."""
    from duet.training.run_sft import build_rows
    recs = [{"id": "a", "history": ["hi"], "profile_self": ["i hike"],
             "profile_partner": ["i run"], "target": "hello"}]
    both = build_rows(apply_profile_scope(recs, True))[0]["__text"]
    self_only = build_rows(apply_profile_scope(recs, False))[0]["__text"]
    assert "PARTNER (them):\n- i run" in both
    assert "PARTNER" not in self_only
    assert both.replace("\n\nPARTNER (them):\n- i run", "") == self_only


# ---------------- PAL samples -> entailment pairs (stage 02c -> 03) ---------

def _pal_samples_and_records():
    records = [
        {"id": "v0", "history": ["hi"], "profile_self": ["i hike"],
         "profile_partner": ["i run"], "target": "gold a"},
        {"id": "v1", "history": ["yo"], "profile_self": ["i code"],
         "profile_partner": [], "target": "gold b"},
    ]
    samples = [
        {"prompt": "PAL-FORMAT PROMPT (ignored)", "golden_response": "gold a",
         "generated_response": "neg a"},
        {"prompt": "PAL-FORMAT PROMPT (ignored)", "golden_response": "gold b",
         "generated_response": "  "},
    ]
    return samples, records


def test_to_pairs_renders_our_format_and_drops_empty():
    samples, records = _pal_samples_and_records()
    rows, dropped = pal_build_pairs(pal_join_samples(samples, records),
                                    both=True)
    assert dropped == 1 and len(rows) == 1
    row = rows[0]
    for key in ("prompt", "chosen", "rejected", "signal", "dialog_id"):
        assert key in row, f"missing pair contract key: {key}"
    assert row["chosen"] == "gold a"
    assert row["rejected"] == "neg a"              # stripped
    assert "PERSONA (you):" in row["prompt"]       # OUR rendering, not PAL's
    assert "PAL-FORMAT" not in row["prompt"]
    assert row["signal"] == "entailment"
    assert row["dialog_id"] == "v0"


def test_to_pairs_aborts_on_misalignment():
    samples, records = _pal_samples_and_records()
    samples[0]["golden_response"] = "SOMETHING ELSE"
    with pytest.raises(SystemExit):
        pal_join_samples(samples, records)
    with pytest.raises(SystemExit):                # length mismatch too
        pal_join_samples(samples[:1], records)


def test_to_pairs_scope():
    samples, records = _pal_samples_and_records()
    samples[1]["generated_response"] = "neg b"
    both_rows, _ = pal_build_pairs(pal_join_samples(samples, records), both=True)
    self_rows, _ = pal_build_pairs(pal_join_samples(samples, records), both=False)
    assert "PARTNER" in both_rows[0]["prompt"]
    assert all("PARTNER" not in r["prompt"] for r in self_rows)
