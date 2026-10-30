"""Component C data: BOTH-perspective expansion of original PERSONA-CHAT.

Source: ParlAI ``*_both_original.txt`` (``data/raw/personachat_original/``) —
the only publicly mirrored PERSONA-CHAT format that carries BOTH speakers'
personas per episode (direct downloads from parl.ai are 403; HF mirrors work).
The PAL-format files we inherit carry only the target speaker's persona, which
is why our §5 records used to have ``profile_partner: []``.

Geometry of one episode — tab-separated pairs (context, target) where c1 is the
MERGED opening turn (partner opener + your opener in one string):

    self:     history=[c1, r1, ..., c_k]     target=r_k   (k=1..N)
    partner:  history=[r1, c2, ..., r_{k-1}] target=c_k   (k=2..N)

Each episode natively covers ONE perspective (verified: no role-swapped twins
in the file), so the partner perspective is synthesized by role swap. Two
turns are NOT supervised for the partner: (a) its opening utterance, fused
into c1 with no recoverable split, and (b) c1 is therefore dropped from partner
histories — which also keeps role parity correct under format_prompt's
even-index=user convention (partner histories must start with the OTHER
speaker's line, r1).

Contract additions over §5 (backward compatible — extra keys):
    perspective          "self" (verbatim) | "partner" (synthesized)
    persona_self_id / persona_partner_id
        sha1 of the persona sentence tuple — gives a stable PROFILE identity,
        enabling profile-level holdout for the critic (plan §6 leakage control)
    dataset_version      "original_both"

Run:  PYTHONPATH=src .venv/bin/python -m duet.bidir.dataset \
          --src data/raw/personachat_original --out data/processed
Owner: Devanshu (Component C). If Kanishk's data pipeline later emits the same
contract, it supersedes this file.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

# Weak persona-relevance label (parity with the old PAL-format converter;
# not a gold annotation — kept because §5 defines the field).
_STOPWORDS = {
    "i", "me", "my", "mine", "we", "our", "you", "your", "he", "she", "they",
    "it", "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "and", "or", "but",
    "to", "of", "in", "on", "at", "for", "with", "as", "this", "that",
    "there", "here", "what", "who", "how", "when", "where", "which", "like",
    "just", "really", "very", "also", "not", "don't", "dont", "can", "could",
    "would", "will", "about", "if",
}


def _content_words(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def weak_persona_label(profile, response):
    """Index of the profile sentence with max lexical overlap, or -1."""
    rw = _content_words(response)
    if not rw:
        return -1
    scores = [len(rw & _content_words(s)) for s in profile]
    if not scores or max(scores) == 0:
        return -1
    return scores.index(max(scores))


def persona_id(profile) -> str:
    """Stable profile identity: hash of the sentence tuple."""
    return hashlib.sha1("\x1f".join(profile).encode("utf-8")).hexdigest()[:12]


def parse_episodes(path):
    """Parse *_both_original.txt into episodes (turn number resets to 1)."""
    episodes, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = re.match(r"^\d+ your persona: (.*)$", line)
            if m:
                if line.startswith("1 ") and cur is not None:
                    episodes.append(cur)
                    cur = None
                if cur is None:
                    cur = {"persona_self": [], "persona_partner": [], "pairs": []}
                cur["persona_self"].append(m.group(1).strip())
                continue
            m = re.match(r"^\d+ partner's persona: (.*)$", line)
            if m:
                if cur is not None:
                    cur["persona_partner"].append(m.group(1).strip())
                continue
            if "\t" in line:                                    # dialogue line
                body = re.sub(r"^\d+ ", "", line)
                fields = body.split("\t")
                if len(fields) >= 2 and fields[0].strip() and fields[1].strip():
                    cur["pairs"].append((fields[0].strip(), fields[1].strip()))
    if cur is not None:
        episodes.append(cur)
    return episodes


def expand_episode(ep, split, i, n_self=0, n_partner=0):
    """Emit §5 records for BOTH perspectives of one episode.

    Ids are file-global (n_self/n_partner are running counters maintained by
    convert_split): self records reuse the legacy sequential ids, partner
    records get a '_p' infix — both unique across the split.
    """
    records = []
    p_self, p_partner = ep["persona_self"], ep["persona_partner"]
    pid_s, pid_p = persona_id(p_self), persona_id(p_partner)

    # --- perspective 1: "self" (verbatim; identical examples + ids to the
    #     legacy PAL-format-derived records, so old/new files can be diffed)
    history = []
    for c, r in ep["pairs"]:
        records.append({
            "id": f"{split}_{n_self:06d}",
            "history": history + [c],
            "profile_self": p_self,
            "profile_partner": p_partner,
            "target": r,
            "persona_labels": [weak_persona_label(p_self, r)],
            "perspective": "self",
            "persona_self_id": pid_s,
            "persona_partner_id": pid_p,
            "episode": i,
            "dataset_version": "original_both",
        })
        n_self += 1
        history += [c, r]

    # --- perspective 2: "partner" (synthesized)
    # c1 is dropped (partner's opener is fused in with no recoverable split);
    # dropping it also keeps role parity right: partner histories start with
    # the other speaker's line so format_prompt labels user/assistant correctly.
    history = []
    for k, (c, r) in enumerate(ep["pairs"]):
        if k >= 1:                                              # target = their turn
            records.append({
                "id": f"{split}_p{n_partner:06d}",
                "history": list(history),
                "profile_self": p_partner,
                "profile_partner": p_self,
                "target": c,
                "persona_labels": [weak_persona_label(p_partner, c)],
                "perspective": "partner",
                "persona_self_id": pid_p,
                "persona_partner_id": pid_s,
                "episode": i,
                "dataset_version": "original_both",
            })
            n_partner += 1
        if k == 0:
            history.append(r)                                   # skip merged opener
        else:
            history += [c, r]
    return records


def convert_split(path, split):
    episodes = parse_episodes(path)
    records, n_self, n_partner = [], 0, 0
    for i, ep in enumerate(episodes):
        out = expand_episode(ep, split, i, n_self, n_partner)
        records += out
        n_self += sum(1 for r in out if r["perspective"] == "self")
        n_partner += sum(1 for r in out if r["perspective"] == "partner")
    return episodes, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/raw/personachat_original")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "valid", "test"])
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        src = Path(a.src) / f"{split}_both_original.txt"
        episodes, records = convert_split(src, split)
        n_self = sum(1 for r in records if r["perspective"] == "self")
        n_partner = len(records) - n_self
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: episodes={len(episodes)} self={n_self} "
              f"partner={n_partner} -> {path}")


if __name__ == "__main__":
    main()
