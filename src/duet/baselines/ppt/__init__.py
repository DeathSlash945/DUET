"""PPT baseline — Pseudo Preference Tuning (Takayama et al., COLING 2025,
pp. 5507-5514; aclanthology.org/2025.coling-main.369).

The Phase-0 second baseline (plan §6): DPO on pseudo-preference pairs built
WITHOUT external NLI data —

    1. SFT the policy (stage 02, our shared run_sft)            -> pi_sft
    2. pseudo-negative: y' ~ pi_sft(y' | p_swapped, x), where p_swapped is
       ONE persona sampled from a DIFFERENT dialogue (paper §3)
    3. pair = (prompt: TRUE persona + history, chosen: gold y, rejected: y')
    4. DPO (paper Appendix B: beta 0.5, lr 1e-6, 3 epochs, eff batch 4)

Paper's own results: qwen2-1.5b +PseudoDPO wins 53.75% vs its SFT, and beats
the DialogueNLI-RL arm — the baseline DUET's critic must contend with (our
reviewer-flagged overlap: PPT already does persona-swapped negatives, so our
critic contribution is the IDENTIFICATION signal, not the swap trick).

Fidelity notes (reproduction deviations, keep in the report):
  - paper generates negatives with vLLM; we use HF generate (same pi_sft),
    sampling params vLLM-default because Appendix B doesn't state them;
  - paper English data = PersonaChat original splits — same files we use.
"""
