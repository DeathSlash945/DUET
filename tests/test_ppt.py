"""PPT baseline tests: persona-swap hygiene + pair assembly. No GPU."""

import random
from types import SimpleNamespace

import pytest

from duet.baselines.ppt.pairs import (assemble_pairs, build_swapped_prompts,
                                      sample_swapped_profile)

A = ["i hike", "i own a red bike"]
B = ["i play chess"]
C = ["i love jazz"]


def _cfg(both=True, seed=42):
    return SimpleNamespace(train=SimpleNamespace(seed=seed),
                           data=SimpleNamespace(both_profiles=both))


def _rec(i, own, partner=None, target="t"):
    return {"id": f"d{i}", "history": [f"h{i}"], "profile_self": own,
            "profile_partner": partner or [], "target": target}


def _three():
    recs = [_rec(1, A), _rec(2, B), _rec(3, C)]
    pool = [(r["id"], r["profile_self"]) for r in recs]
    return recs, pool


def test_swap_never_picks_own_persona():
    recs, pool = _three()
    for r in recs:
        for _ in range(20):
            got = sample_swapped_profile(r, pool, random.Random(0))
            assert tuple(got) != tuple(r["profile_self"])


def test_swap_pool_excludes_shared_persona_dialogues():
    """Two dialogues sharing a persona must never swap onto each other."""
    recs = [_rec(1, A), _rec(2, list(A)), _rec(3, B)]
    pool = [(r["id"], r["profile_self"]) for r in recs]
    for _ in range(20):
        got = sample_swapped_profile(recs[0], pool, random.Random(1))
        assert tuple(got) == tuple(B)   # only foreign persona available


def test_empty_swap_pool_raises():
    rec = _rec(1, A)
    with pytest.raises(ValueError):
        sample_swapped_profile(rec, [("d1", A)], random.Random(0))


def test_build_swapped_prompts_deterministic_and_truly_swapped():
    recs, pool = _three()
    _, p1 = build_swapped_prompts(recs, pool, _cfg(seed=7))
    _, p2 = build_swapped_prompts(recs, pool, _cfg(seed=7))
    assert p1 == p2                                    # same seed -> same swaps
    for r, p in zip(recs, p1):
        assert r["history"][0] in p                    # same dialogue history x_i
        for line in r["profile_self"]:
            assert line not in p                       # own persona swapped OUT


def test_negative_prompt_partner_follows_scope():
    recs = [_rec(1, A, partner=C)]
    pool = [("d1", A), ("d2", B)]
    _, both_prompts = build_swapped_prompts(recs, pool, _cfg(both=True))
    _, self_prompts = build_swapped_prompts(recs, pool, _cfg(both=False))
    assert "PARTNER" in both_prompts[0]
    assert "PARTNER" not in self_prompts[0]


def test_pairs_true_prompt_gold_chosen():
    """The DPO row: prompt = TRUE persona + history; chosen = gold; rejected =
    swapped-persona generation (paper §3: l_i = (p_i, x_i, y_i, y'_neg))."""
    recs = [_rec(1, A, target="gold response")]
    rows = assemble_pairs(recs, ["<NEG>"], {"both_profiles": True})
    row = rows[0]
    assert row["chosen"] == "gold response"
    assert row["rejected"] == "<NEG>"
    for line in A:
        assert line in row["prompt"]                   # TRUE persona intact
    assert row["signal"] == "pseudo_preference"
    assert row["dialog_id"] == "d1"


def test_pairs_scope_selfonly_strips_partner():
    rows = assemble_pairs([_rec(1, A, partner=C)], ["neg"], {"both_profiles": False})
    assert "PARTNER" not in rows[0]["prompt"]


def test_pairs_require_required_keys():
    rows = assemble_pairs([_rec(1, A)], ["neg"], None)
    for key in ("prompt", "chosen", "rejected"):
        assert key in rows[0], f"missing DPO contract key: {key}"
