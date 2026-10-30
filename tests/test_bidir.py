"""CPU-only tests for Component C data: both-perspective expansion."""

from duet.bidir.dataset import (
    expand_episode, parse_episodes, persona_id, weak_persona_label)
from duet.training.prompting import format_prompt

EP = {
    "persona_self": ["i like music", "i play guitar"],
    "persona_partner": ["i love dogs", "i work as an engineer"],
    "pairs": [
        ("hello there friend , i like music", "hi ! do you like dogs ?"),
        ("yes i love dogs very much", "i play guitar on weekends"),
        ("cool , guitar is fun to hear", "i also work as an engineer"),
    ],
}


def test_expand_counts_and_ids():
    recs = expand_episode(EP, "train", 0, n_self=100, n_partner=50)
    self_r = [r for r in recs if r["perspective"] == "self"]
    partner_r = [r for r in recs if r["perspective"] == "partner"]
    assert len(self_r) == 3 and len(partner_r) == 2      # N + (N-1)
    assert [r["id"] for r in self_r] == ["train_000100", "train_000101",
                                         "train_000102"]
    assert [r["id"] for r in partner_r] == ["train_p000050", "train_p000051"]


def test_self_perspective_geometry():
    recs = expand_episode(EP, "t", 0)
    first = recs[0]
    assert first["history"] == [EP["pairs"][0][0]]       # c1 only
    assert first["target"] == EP["pairs"][0][1]
    third = recs[2]
    assert third["history"] == [EP["pairs"][0][0], EP["pairs"][0][1],
                                EP["pairs"][1][0], EP["pairs"][1][1],
                                EP["pairs"][2][0]]       # c1 r1 c2 r2 c3
    assert third["target"] == EP["pairs"][2][1]          # r3
    assert third["profile_self"] == EP["persona_self"]
    assert third["profile_partner"] == EP["persona_partner"]


def test_partner_perspective_geometry_and_parity():
    recs = [r for r in expand_episode(EP, "t", 0)
            if r["perspective"] == "partner"]
    first, second = recs
    # merged opener c1 is dropped; first supervised turn is c2 after r1
    assert first["history"] == [EP["pairs"][0][1]]       # r1 (other speaker)
    assert first["target"] == EP["pairs"][1][0]          # c2
    assert second["history"] == [EP["pairs"][0][1], EP["pairs"][1][0],
                                 EP["pairs"][1][1]]      # r1 c2 r2
    assert second["target"] == EP["pairs"][2][0]         # c3
    # profiles swapped
    assert first["profile_self"] == EP["persona_partner"]
    assert first["profile_partner"] == EP["persona_self"]
    # role parity under format_prompt: even index = user = other speaker's line
    for r in recs:
        lines = format_prompt(r["history"], r["profile_self"],
                              r["profile_partner"]).splitlines()
        dialogue = [l.split(": ", 1)[1] for l in lines
                    if l.startswith(("user: ", "assistant: "))]
        assert dialogue == r["history"]                  # every line round-trips
        for idx, utt in enumerate(r["history"]):
            role = "user" if idx % 2 == 0 else "assistant"
            assert dialogue[idx] == utt                  # ordering preserved


def test_persona_ids_swap_and_stability():
    recs = expand_episode(EP, "t", 0)
    for r in recs:
        if r["perspective"] == "self":
            assert r["persona_self_id"] == persona_id(EP["persona_self"])
            assert r["persona_partner_id"] == persona_id(EP["persona_partner"])
        else:
            assert r["persona_self_id"] == persona_id(EP["persona_partner"])
            assert r["persona_partner_id"] == persona_id(EP["persona_self"])
    assert persona_id(EP["persona_self"]) != persona_id(EP["persona_partner"])


def test_parse_episodes(tmp_path):
    lines = []
    for i, s in enumerate(EP["persona_self"], 1):
        lines.append(f"{i} your persona: {s}")
    for i, p in enumerate(EP["persona_partner"], 1):
        lines.append(f"{i} partner's persona: {p}")
    for j, (c, r) in enumerate(EP["pairs"], 1):
        lines.append(f"{j} {c}\t{r}\t\tdistractor|pool")
    lines.append("1 your persona: second ep opener.")
    lines.append("1 partner's persona: second ep partner.")
    lines.append("1 hi there\thello back\t\td1|d2")
    f = tmp_path / "toy.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    eps = parse_episodes(f)
    assert len(eps) == 2
    assert eps[0]["persona_self"] == EP["persona_self"]
    assert eps[0]["persona_partner"] == EP["persona_partner"]
    assert eps[0]["pairs"] == EP["pairs"]
    assert eps[1]["pairs"] == [("hi there", "hello back")]


def test_weak_label():
    assert weak_persona_label(["i love dogs", "i code in python"], "i love dogs") == 0
    assert weak_persona_label(["i love dogs"], "completely unrelated text") == -1
