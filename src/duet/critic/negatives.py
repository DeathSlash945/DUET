"""Candidate-set construction for the critic (Component A negative sampling).

The critic's shortcut risk (guide §"big risk"): matching responses to profiles
on shared WORDS rather than meaning. Controls, all here:
  - hard distractors: decoys sharing surface vocabulary with the TRUE profile
    (highest Jaccard overlap), so word-matching is ambiguous;
  - copy probes: verbatim profile text as the response -> labeled NO_MATCH,
    so copying is penalized, not rewarded;
  - generic replies: canned filler -> NO_MATCH (genericness is unidentifiable,
    so it automatically scores low — no separate diversity term needed).

Candidate layout for every example: [true, distractor_1..k, NO_MATCH_SLOT],
with the true profile's position shuffled among the profile slots and the
no-match slot always LAST.
"""

import random

# Fixed sentinel candidate ("no matching profile" slot from the old prototype).
NO_MATCH_TEXT = "No matching personal description."

# Canned replies that carry no persona information (probe bank; sampled per
# record so probes and real examples stay class-balanced).
GENERIC_REPLIES = [
    "That is interesting, tell me more.",
    "I see. What else is going on with you?",
    "That sounds great! How is your day going?",
    "Wow, that is cool. I like talking with you.",
    "Oh really? That is nice to hear.",
    "I am not sure what to say, but I am glad we are chatting.",
    "Haha yes. Anyway, what do you like to do?",
    "Nice! I hope you are having a good day.",
]


def jaccard(a, b) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def render_profile(profile) -> str:
    return " ".join(f"{i + 1}.{s.strip()}" for i, s in enumerate(profile))


def copy_probe(profile, rng: random.Random) -> str:
    """A verbatim profile response: full concatenation or a single sentence."""
    if len(profile) > 1 and rng.random() < 0.5:
        return rng.choice(list(profile)).strip()
    return " ".join(s.strip() for s in profile)


def sample_distractors(true_profile, pool, n_hard, n_random, rng: random.Random,
                       exclude_ids=()):
    """Pick distractor profiles: `n_hard` highest-Jaccard decoys + `n_random`
    random ones. `pool` is a list of (id, profile); ids in exclude_ids (the
    record's own profile) are never chosen; duplicates are dropped."""
    true_text = render_profile(true_profile)
    eligible = [(pid, prof) for pid, prof in pool if pid not in set(exclude_ids)
                and render_profile(prof) != true_text]
    ranked = sorted(eligible, key=lambda ip: jaccard(true_text, render_profile(ip[1])),
                    reverse=True)
    chosen = ranked[:n_hard]
    remaining = [ip for ip in eligible if ip not in chosen]
    chosen += rng.sample(remaining, min(n_random, len(remaining)))
    return [prof for _, prof in chosen]


def build_example(record, pool, cfg, rng: random.Random) -> dict:
    """One REAL training/validation example: the record's gold response,
    traceable to its true profile."""
    n_hard = round(cfg.n_distractors * cfg.hard_fraction)
    n_random = cfg.n_distractors - n_hard
    distractors = sample_distractors(record["profile_self"], pool, n_hard, n_random,
                                     rng, exclude_ids=(record["id"],))
    profiles = [record["profile_self"]] + list(distractors)
    true_idx = rng.randrange(len(profiles))          # shuffle: no positional shortcut
    profiles.insert(true_idx, profiles.pop(0))
    candidates = [render_profile(p) for p in profiles] + [NO_MATCH_TEXT]
    return {"dialog_id": record["id"], "response": record["target"],
            "candidates": candidates, "label": true_idx, "category": "real"}


def build_probes(record, pool, cfg, rng: random.Random) -> list:
    """COPY and GENERIC probes for one record (both -> no-match slot)."""
    ex = build_example(record, pool, cfg, rng)       # reuse its candidate set
    slot = len(ex["candidates"]) - 1
    probes = [{**ex, "response": copy_probe(record["profile_self"], rng), "label": slot,
               "category": "copy"}]
    probes.append({**ex, "response": rng.choice(GENERIC_REPLIES), "label": slot,
                   "category": "generic"})
    return probes
