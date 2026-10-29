"""CPU-only tests for Component A (critic): candidates, model I/O, pairs, refresh."""

import json
import random

import pytest
import torch
from transformers import BertConfig, BertForMultipleChoice, BertTokenizerFast

from duet.critic.model import encode_examples, p_true, score_examples
from duet.critic.negatives import (
    GENERIC_REPLIES, NO_MATCH_TEXT, build_example, build_probes, copy_probe,
    jaccard, render_profile, sample_distractors)
from duet.critic.refresh import build_refresh_examples
from duet.critic.train import build_examples
from duet.training.common import to_ns

# --------------------------------------------------------------------------- fixtures

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + \
        ["i", "am", "a", "student", "who", "loves", "music", "coding", "dogs",
         "cats", "pizza", "hiking", "movies", "travel", "coffee", "sports",
         "play", "guitar", "the", "on", "weekends", "with", "friends", "my",
         "family", "work", "as", "engineer", "eat", "like", "very", "much",
         "no", "matching", "personal", "description", "interesting", "tell",
         "more", "day", "going", "cool", "glad", "we", "are", "chatting", "##s", "##ing"]

REC = {
    "id": "d1",
    "history": [["hi", "hello"], ["i am a student", "nice"]],
    "profile_self": ["i am a student", "i love music", "i play guitar on weekends"],
    "profile_partner": ["i like cats", "i work as an engineer"],
    "target": "i play guitar with my friends on weekends",
}
POOL = [(f"p{i}", [f"i love dogs", f"i eat pizza", f"number {i} thing"])
        for i in range(10)] + [(REC["id"], REC["profile_self"])]


@pytest.fixture(scope="module")
def tiny_factory(tmp_path_factory):
    """Random-init BERT-MC + tokenizer saved to disk (no downloads).

    Returns a factory so mutating tests (train_from) get their own copy.
    """
    d = tmp_path_factory.mktemp("tiny_critic")
    vocab_file = d / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB))
    tok = BertTokenizerFast(str(vocab_file), unk_token="[UNK]",
                            pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]", mask_token="[MASK]")
    tok.save_pretrained(d)
    cfg = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=128)
    BertForMultipleChoice(cfg).save_pretrained(d)

    def make():
        from transformers import AutoModelForMultipleChoice, AutoTokenizer
        return (str(d), AutoModelForMultipleChoice.from_pretrained(d),
                AutoTokenizer.from_pretrained(d))
    return make


@pytest.fixture
def tiny(tiny_factory):
    return tiny_factory()


CFG = to_ns({
    "candidates": {"n_distractors": 3, "hard_fraction": 0.67},
    "train": {"batch_size": 4, "eval_batch_size": 8, "epochs": 1, "lr": 1e-4,
              "weight_decay": 0.0, "warmup_ratio": 0.0, "grad_clip": 1.0,
              "bf16": False, "seed": 42},
    "data": {"max_seq_len": 64},
    "gate": {"real_acc_min": 0.60, "copy_to_slot_min": 0.80,
             "generic_to_slot_min": 0.80},
})

# --------------------------------------------------------------------------- negatives

def test_jaccard():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b", "c d") == 0.0
    assert 0 < jaccard("a b c", "b c d") < 1


def test_sample_distractors_excludes_self_and_dedupes():
    rng = random.Random(0)
    true = REC["profile_self"]
    distractors = sample_distractors(true, POOL, 2, 1, rng,
                                     exclude_ids=(REC["id"],))
    assert len(distractors) == 3
    assert true not in distractors
    texts = [render_profile(p) for p in distractors]
    assert len(set(texts)) == 3


def test_hard_distractors_prefer_high_overlap():
    shared = ["i am a student", "i love music"]     # overlaps true profile
    pool = [("far", ["pizza pizza pizza"]), ("near", shared)] + \
           [(f"x{i}", ["unrelated words"]) for i in range(5)]
    d = sample_distractors(REC["profile_self"], pool, 1, 0, random.Random(0))
    assert d[0] == shared                            # hard negative wins


def test_build_example_layout():
    ex = build_example(REC, POOL, CFG.candidates, random.Random(3))
    assert len(ex["candidates"]) == 5                # true + 3 distractors + slot
    assert ex["candidates"][-1] == NO_MATCH_TEXT
    assert ex["label"] in range(4)                   # never the slot for real
    assert ex["category"] == "real"
    assert render_profile(REC["profile_self"]) in ex["candidates"]


def test_probes_labeled_to_slot():
    rng = random.Random(1)
    for probe in build_probes(REC, POOL, CFG.candidates, rng):
        assert probe["label"] == 4                   # no-match slot (last)
        assert probe["category"] in ("copy", "generic")


def test_copy_probe_is_verbatim():
    rng = random.Random(7)
    probe = copy_probe(REC["profile_self"], rng)
    assert any(s in probe for s in REC["profile_self"])   # verbatim sentence
    assert probe in GENERIC_REPLIES or all(
        w in " ".join(VOCAB) for w in probe.split())      # no novel content


def test_build_examples_balanced():
    exs = build_examples([REC], POOL, CFG, random.Random(0))
    assert len(exs) == 3                             # real + copy + generic
    assert sorted({e["category"] for e in exs}) == ["copy", "generic", "real"]

# --------------------------------------------------------------------------- model I/O

def test_encode_and_score_shapes(tiny):
    _, model, tok = tiny
    exs = build_examples([REC], POOL, CFG, random.Random(0))
    input_ids, attention_mask, labels = encode_examples(tok, exs, 64, "cpu")
    assert input_ids.shape == (3, 5, input_ids.shape[-1])
    assert attention_mask.shape == input_ids.shape
    assert labels.tolist() == [e["label"] for e in exs]

    probs = score_examples(model, tok, exs, "cpu", bf16=False)
    assert probs.shape == (3, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-4)


def test_p_true_picks_label_mass():
    probs = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]])
    got = p_true(probs, torch.tensor([0, 2])).tolist()
    assert got == pytest.approx([0.7, 0.6], abs=1e-6)

# --------------------------------------------------------------------------- train / gate

def test_train_report_and_gate(tiny, tmp_path):
    from duet.critic.train import train_from, evaluate
    path, model, tok = tiny
    train_ex = build_examples([REC, {**REC, "id": "d2"}], POOL, CFG,
                              random.Random(0))
    eval_ex = build_examples([REC], POOL, CFG, random.Random(1))
    report = train_from(model, tok, train_ex, eval_ex, CFG, tmp_path / "ck", "cpu")
    for key in ("overall_acc", "real_acc", "copy_to_slot_acc",
                "generic_to_slot_acc", "chance", "gate"):
        assert key in report
    assert report["chance"] == pytest.approx(0.2)
    assert (tmp_path / "ck" / "final").exists()
    assert (tmp_path / "ck" / "LATEST").is_symlink()
    # random-init tiny model: gate should FAIL (not be vacuously true)
    assert report["gate"]["passed"] in (True, False)

# --------------------------------------------------------------------------- rewards

def test_build_pairs_gold_vs_worst(tiny, tmp_path):
    from duet.critic.rewards import build_pairs
    _, model, tok = tiny
    records = {REC["id"]: REC}
    generations = [
        {"dialog_id": "d1", "response": "the cats and dogs"},
        {"dialog_id": "d1", "response": "hello my friend hello"},
    ]
    cfg = to_ns({**CFG.__dict__, "pool": POOL})
    pairs = build_pairs(records, generations, cfg, model, tok, "cpu", seed=0)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["chosen"] == REC["target"]            # gold anchor
    assert pair["rejected"] in {g["response"] for g in generations}
    assert pair["signal"] == "identification"
    assert pair["n_samples"] == 2
    assert "prompt" in pair and len(pair["prompt"]) > 0

# --------------------------------------------------------------------------- refresh

def test_refresh_examples_include_policy_outputs():
    gens = [{"dialog_id": "d1", "response": "i am a student and i love music"},
            {"dialog_id": "d1", "response": "i play guitar"}]
    exs = build_refresh_examples([REC], POOL, CFG, random.Random(0), gens)
    real = [e for e in exs if e["category"] == "real"]
    assert len(real) == 3                            # gold + 2 policy responses
    assert {e["response"] for e in real} >= {REC["target"]} | \
           {g["response"] for g in gens}
    # gold-only eval stays a fixed yardstick
    eval_exs = build_refresh_examples([REC], POOL, CFG, random.Random(0), None)
    assert len([e for e in eval_exs if e["category"] == "real"]) == 1
