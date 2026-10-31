"""PRELIMINARY scorer over generation dumps — interim pass until Aparajith's
eval/harness.py lands (contract: project_plan.md §5, consumption only).

    python -m duet.eval.prelim [dump.jsonl ...] [--out reports/prelim.md]

Aggregates per `condition` over the dump rows {run_id, model, condition,
dialog_id, profile_self, profile_partner, response} and prints a Markdown
table ready to paste into report/main.tex §Results:

  copy@1      fraction of response TOKENS appearing verbatim in profile_self
  copy@4      fraction of response 4-GRAMS appearing verbatim in profile_self
              (both = the plan's copy-rate metric; paraphrase evades them by
              design, flagged as a limitation in the report)
  self_cov    profile_self content words expressed in the response
  partner_cov same over profile_partner (0 unless the dump is bidirectional)
  resp_len    mean response tokens

Identifiability (critic-based) is NOT computed here — it comes from the
critic's own held-out report (scripts/04_train_critic.sh) and later the
harness, so no metric ever reuses the pair-building signal.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from ..training.common import read_jsonl

STOP = set("""a an and are as at be but by for from has have i if in is it its
me my of on or our so that the their them they this to was we were what when
who will with you your am not no do does did don didnt just really very too
can could would should about after all also any been before being between
both each few more most other some such than then there these through under
up""".split())

_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text):
    return _TOKEN.findall(text.lower())


def content_words(text):
    return [w for w in tokens(text) if w not in STOP and len(w) > 2]


def ngrams(words, n):
    return tuple(zip(*[words[i:] for i in range(n)])) if len(words) >= n else []


def score_row(row):
    resp = row.get("response", "") or ""
    self_txt = " ".join(row.get("profile_self") or [])
    partner_txt = " ".join(row.get("profile_partner") or [])
    rt = tokens(resp)
    st, pt = tokens(self_txt), tokens(partner_txt)

    def copy_at(k):
        if len(rt) < k:
            return None
        prof = set(ngrams(st, k))
        grams = ngrams(rt, k)
        return sum(1 for g in grams if g in prof) / len(grams)

    sc, pc = set(content_words(self_txt)), set(content_words(partner_txt))
    rc = set(content_words(resp))
    return {
        "copy@1": copy_at(1),
        "copy@4": copy_at(4),
        "self_cov": (len(rc & sc) / len(sc)) if sc else None,
        "partner_cov": (len(rc & pc) / len(pc)) if pc else None,  # None: no
        "resp_len": len(rt) if rt else None,                      # partner seen
    }


def aggregate(dumps):
    groups = defaultdict(list)
    for path in dumps:
        for row in read_jsonl(path):
            groups[row.get("condition", "?")].append(score_row(row))
    out = {}
    for cond, rows in groups.items():
        stats = {}
        for key in rows[0]:
            vals = [r[key] for r in rows if r[key] is not None]
            stats[key] = sum(vals) / len(vals) if vals else None
        stats["n"] = len(rows)
        out[cond] = stats
    return out


def to_markdown(agg):
    cols = ["n", "copy@1", "copy@4", "self_cov", "partner_cov", "resp_len"]
    head = "| condition | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [head, sep]
    for cond in sorted(agg):
        s = agg[cond]

        def fmt(v):
            if v is None:
                return "--"
            return str(v) if isinstance(v, int) else f"{v:.3f}"

        lines.append(f"| {cond} | " + " | ".join(fmt(s.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", help="generations.jsonl files")
    ap.add_argument("--out", default=None, help="also write Markdown here")
    a = ap.parse_args()

    missing = [d for d in a.dumps if not Path(d).exists()]
    if missing:
        raise SystemExit(f"missing dumps: {missing}")
    md = to_markdown(aggregate(a.dumps))
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md + "\n", encoding="utf8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
