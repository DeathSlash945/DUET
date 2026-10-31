"""Preliminary scorer tests — interim metrics until Aparajith's harness."""

from duet.eval.prelim import aggregate, score_row, to_markdown


def _row(resp, self_p, partner_p, cond="sft"):
    return {"condition": cond, "response": resp, "profile_self": self_p,
            "profile_partner": partner_p}


def test_copy_metrics_verbatim_vs_novel():
    copied = score_row(_row("i read twenty books a year",
                            ["i read twenty books a year.", "i code."], []))
    assert copied["copy@1"] == 1.0
    assert copied["copy@4"] == 1.0
    novel = score_row(_row("totally unrelated answer here",
                           ["i read twenty books"], []))
    assert novel["copy@1"] == 0.0
    assert novel["copy@4"] == 0.0


def test_short_response_copy4_is_none():
    r = score_row(_row("hi there", ["i code."], []))
    assert r["copy@4"] is None          # fewer than 4 tokens -> "--"


def test_coverage_uses_content_words_only():
    r = score_row(_row("i love hiking and my dog",
                       ["i love hiking.", "i own a red bike"], []))
    # self content = {love, hiking, own, red, bike}; response hits 2
    assert abs(r["self_cov"] - 0.4) < 1e-9
    assert r["partner_cov"] is None     # no partner in context -> "--"


def test_partner_cov_active_when_partner_present():
    r = score_row(_row("i run marathons with you",
                       ["i code."], ["i run marathons."]))
    assert r["partner_cov"] > 0.5


def test_aggregate_groups_and_markdown(tmp_path):
    import json
    d1 = tmp_path / "a.jsonl"
    d2 = tmp_path / "b.jsonl"
    d1.write_text(json.dumps(_row("i code daily", ["i code."], [], "sft_selfonly")) + "\n")
    d2.write_text(json.dumps(_row("i code daily", ["i code."], [], "sft")) + "\n")
    agg = aggregate([d1, d2])
    assert set(agg) == {"sft", "sft_selfonly"}
    assert all(v["n"] == 1 for v in agg.values())
    md = to_markdown(agg)
    assert "| sft |" in md and "copy@4" in md
