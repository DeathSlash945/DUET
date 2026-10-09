"""Copy rate: fraction of generated responses that reuse a long n-gram span from the profile.
Works on decode.py style output: a JSON list of {"prompt", "generated_response"}.
The profile is read back out of the prompt text.

Usage: python copy_rate.py generated/your_output.json [--n 4]
"""
import json, re, sys, argparse


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def ngrams(w, n):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def profile_from_prompt(prompt):
    m = re.search(r"personal descriptions set: '''(.*?)'''\.", prompt, re.S)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="fraction of response n-grams found in profile to count as copy")
    a = ap.parse_args()

    data = json.load(open(a.path, encoding="utf-8"))
    copied, scored, overlaps = 0, 0, []
    for item in data:
        prof = profile_from_prompt(item["prompt"])
        rg = ngrams(words(item["generated_response"]), a.n)
        if not prof or not rg:
            continue
        ov = len(rg & ngrams(words(prof), a.n)) / len(rg)
        overlaps.append(ov)
        scored += 1
        copied += ov >= a.thresh
    print(f"responses scored: {scored}")
    print(f"copy rate (>= {a.thresh:.0%} of {a.n}-grams in profile): {copied / max(scored, 1):.3f}")
    print(f"mean {a.n}-gram overlap: {sum(overlaps) / max(len(overlaps), 1):.3f}")


if __name__ == "__main__":
    main()
