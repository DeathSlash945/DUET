"""Pure-function tests for the evaluation harness (no GPU, no model loads)."""

from duet.eval.harness import (copy_rate, is_self_only, partner_recall,
                               self_mention, unsupported_claim)

PROF = ["i like to remodel homes.", "i like to go hunting."]


def test_copy_rate_verbatim_vs_unrelated():
    assert copy_rate("i like to remodel homes.", PROF, 1) == 1.0
    assert copy_rate("i enjoy quantum physics.", PROF, 3) == 0.0


def test_unsupported_claim_flags_foreign_persona():
    assert unsupported_claim("I love skateboarding.", PROF) is True
    assert unsupported_claim("I like to remodel homes.", PROF) is False
    # non first-person sentences are never counted
    assert unsupported_claim("That sounds fun.", PROF) is False


def test_partner_recall():
    partner = ["i like canning and whittling.", "my favorite color is blue."]
    assert partner_recall("we should try canning", partner) == 0.5
    assert partner_recall("hello there", partner) == 0.0
    assert partner_recall("anything", []) is None


def test_self_mention():
    assert self_mention("i like to remodel homes", PROF) == 0.5
    assert self_mention("no overlap at all", PROF) == 0.0


def test_self_only_condition_flag():
    assert is_self_only("sft_selfonly")
    assert is_self_only("sft_self_only")
    assert not is_self_only("sft")
    assert not is_self_only("dpo_critic")
