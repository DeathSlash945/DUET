"""The DUET policy prompt format — single source of truth.

Every component that turns a dialogue record into a policy prompt MUST import
from here: SFT, DPO pair builders, steering evaluation, bidirectional runs, and
the eval harness. Changing this file changes every experiment, so changes need
a team sync first (see project_plan.md §5).
"""


def format_prompt(history, profile_self, profile_partner=None) -> str:
    """Render one policy prompt from a §5 dialogue record.

    - Even history indices are the partner (user), odd are the policy (assistant).
    - The partner section is emitted only when profile_partner is non-empty,
      which is what makes baseline (self-only) and bidirectional (both) runs
      differ by data alone, not by prompt plumbing.
    """
    lines = ["PERSONA (you):"]
    lines += [f"- {s}" for s in profile_self]
    if profile_partner:
        lines.append("")
        lines.append("PARTNER (them):")
        lines += [f"- {s}" for s in profile_partner]
    lines.append("")
    lines.append("DIALOGUE:")
    for i, utt in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        lines.append(f"{role}: {utt}")
    lines.append("assistant:")
    return "\n".join(lines)


def format_sft_text(record) -> str:
    """Full SFT training string: prompt + gold target + eos."""
    prompt = format_prompt(
        record["history"], record["profile_self"], record.get("profile_partner")
    )
    return prompt + " " + record["target"]
