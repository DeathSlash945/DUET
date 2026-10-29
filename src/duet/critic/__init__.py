"""Component A — adversarial persona critic (Devanshu).

Identification-based reward replacing the entailment signal: a small encoder
classifier guesses which profile produced a response; the policy is rewarded
when the TRUE profile can be identified. Hard negative sampling (shared
vocabulary, copy probes, generic replies) blocks the lexical-shortcut failure
mode; fixed-interval refresh blocks stale-critic exploitation.

Modules: model (architecture), negatives (candidate construction), train
(training + isolation validation gate), refresh (adversarial refresh loop),
rewards (identification signal -> preference pairs).
"""

__all__ = ["model", "negatives", "train", "refresh", "rewards"]
