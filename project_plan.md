# DUET — Project Plan

**Team Phronesis** · Devanshu Sabharwal, Kanishk Nandeshwar, S Aparajith
Machine Learning Course Project · 12-week plan
Base implementation: [kylokano/PAL](https://github.com/kylokano/PAL) · Repo: [DeathSlash945/DUET](https://github.com/DeathSlash945/DUET)

> Source of truth: the revised SOP (`ProjectFilesSubmitted/duet_sop.tex`).
> This plan turns that SOP into who-builds-what, in what order, with decision gates.

---

## 1. One-paragraph summary

Persona-grounded dialogue models are trained so responses *do not contradict* a short profile, using an NLI/entailment score as the reward. That reward is gameable: verbatim copying and bland generic text both pass. DUET replaces the signal — instead of asking "does this response agree with the profile?" we ask "could an observer identify **which** profile produced this response?" — and tests three claims on PERSONA-CHAT with a 1.5B-class LoRA-tuned policy: (A) an identification-based critic beats entailment as a training signal, (B) persona attributes are separable directions in activation space and can be composed, (C) a model trained with **both** speakers' profiles uses the partner's persona. Evaluation is owned independently of training.

## 2. The three claims (what we are actually testing)

1. **The supervision signal is wrong.** An identification-based critic should beat an entailment-based reward under matched conditions (Component A).
2. **The representation is too coarse.** Persona attributes should be separable directions in activation space, and we ask whether they compose (Component B).
3. **The scope is too narrow.** A dialogue model should use the partner's profile, not only its own (Component C).

**Out of scope (do not propose these):** multilingual personas, cross-session memory, persona induction from history, model scales beyond our hardware.

**Not ours (baselines/inherited — never claim as novel):** entailment-based pair construction (PAL/PPT), decoding-time logit mixing (PAD), the identification *idea* itself (descends from phredGAN / P²Bot), steering vectors for global traits (RePE/ITI/CAA/PersonaVectors), contradiction & drift measurement framings (prior work).

## 3. Team & ownership

One owner per training component; evaluation is owned by someone who does **not** own a training component (no grading your own homework).

| Member | Owns | Does **not** own |
|---|---|---|
| **Devanshu Sabharwal** | **Component A** — adversarial persona critic: architecture, negative sampling, adversarial refresh loop, critic-guided preference optimization run. **Component C** — bidirectional training (default owner; per SOP reassigned at Phase 3 start to whoever has capacity). **Stretch** — token-level credit assignment. | Evaluation metrics (Aparajith's). |
| **Kanishk Nandeshwar** | **Data pipeline** (retains *both* speakers' profiles), **Component B** — attribute direction extraction, layer & magnitude sweeps, composition experiments, log-probability persona sensitivity metrics. | Critic training, final eval harness. |
| **S Aparajith** | **Evaluation in full** — partner-persona suite, self-consistency ledger, drift curves, two independently seeded NLI checkpoints, Original→Revised robustness experiment, critic overoptimization study. | Any training component. |

**Interfaces between owners are contracts (Section 5)** — they let all three work in parallel from week 1.

## 4. Repository plan — from scratch

The old repo content (copied PAL files + early experiments) is discarded. Target structure:

```
DUET/
├── README.md                  # what DUET is, install, reproduce, results
├── project_plan.md            # this file
├── requirements.txt
├── .gitignore                 # data/, outputs/, checkpoints/, caches/, *.pt
├── configs/                   # one YAML per experiment — every run is config-driven
│   ├── sft.yaml
│   ├── dpo_entailment.yaml    # baseline signal (PAL/PPT-style)
│   ├── dpo_critic.yaml        # Component A
│   ├── steer.yaml             # Component B
│   └── bidir.yaml             # Component C
├── src/duet/
│   ├── data/                  # Kanishk — PERSONA-CHAT pipeline
│   │   ├── persona_chat.py    # loads data, KEEPS BOTH profiles
│   │   ├── attributes.py      # profile → per-attribute segments (for B)
│   │   └── schemas.py         # shared dataclasses / JSON schemas (contract)
│   ├── baselines/
│   │   ├── pal/               # reproduced from kylokano/PAL
│   │   └── ppt/               # PPT reimplementation from paper
│   ├── critic/                # Devanshu — Component A
│   │   ├── model.py           # encoder-based identification classifier
│   │   ├── train.py           # critic training + isolation validation
│   │   ├── refresh.py         # fixed-interval adversarial refresh
│   │   └── rewards.py         # identification signal → preference pairs
│   ├── steering/              # Kanishk — Component B
│   │   ├── extract.py         # paired-activation mean differences per layer
│   │   ├── sweep.py           # layer × magnitude sweeps
│   │   └── apply.py           # forward hooks, per-attribute magnitude
│   ├── bidir/                 # Devanshu — Component C (both profiles in context)
│   ├── training/              # shared SFT/DPO wrappers over TRL + PEFT
│   └── eval/                  # Aparajith — evaluation harness
│       ├── harness.py         # runs ALL metrics from one generations file
│       ├── identifiability.py # held-out critic identification accuracy
│       ├── nli.py             # two independently seeded NLI checkpoints
│       ├── copy_rate.py       # n-gram overlap with profile
│       ├── sensitivity.py     # log-prob delta, true vs perturbed profile
│       ├── partner.py         # recall / contradiction / questions
│       ├── ledger.py          # self-consistency ledger
│       ├── drift.py           # expression decay vs dialogue depth
│       └── overopt.py         # frozen-critic overoptimization study
├── scripts/                   # runnable entry points, one per stage
│   ├── 00_download_data.sh
│   ├── 01_build_dataset.sh
│   ├── 02_sft.sh
│   ├── 03_dpo.sh
│   ├── 04_train_critic.sh
│   ├── 05_extract_directions.sh
│   └── 90_run_eval.sh
├── tests/                     # schema tests + smoke tests on a data sample
├── notebooks/                 # exploration only — never the source of truth
├── reports/                   # results tables, figures, weekly notes
└── data/                      # GITIGNORED: raw/ processed/ caches/
```

**Working agreements**

- **Branches:** `main` = working state; one feature branch per member per task (`devanshu/critic-train`, `zala/data-pipeline`, `sumagna/eval-harness`). No direct pushes to `main`; one teammate reviews before merge.
- **Everything is config-driven:** a run must be reproducible from `configs/*.yaml` + a script, never from edited notebook state.
- **Commit style:** short imperative summary + what/why in body (`add distractor sampling with lexical-overlap buckets`).
- **Raw data and checkpoints never enter git** (`.gitignore` enforces it).
- **Weekly note** in `reports/weekly/` — 5 lines: what ran, what the numbers said, what's next, blockers.

## 5. Shared contracts (so the three workstreams never block each other)

All defined in `src/duet/data/schemas.py` — one file, owned by Kanishk, consumed by everyone.

1. **Processed dialogue record** (`data/processed/{split}.jsonl`) — produced by the pipeline, consumed by SFT, critic, steering, and eval:
   ```json
   {
     "id": "train_000123",
     "history": ["utterance", "..."],
     "profile_self": ["attr sentence", "..."],
     "profile_partner": ["attr sentence", "..."],
     "target": "gold response",
     "persona_labels": [2],
     "dataset_version": "original"          // "original" | "revised"
   }
   ```
   *Both profiles are always present — this is what makes Component C possible with zero new data collection.*
2. **Critic candidate sets** — produced and consumed *in-memory* by `critic/negatives.py` (`build_example`/`build_probes`), so no intermediate file is needed: response, true profile, `k` distractor profiles (lexical-overlap), no-match slot, and `copy`/`generic` probe flags. DPO pairs land in the §5 pair contract below via `critic/rewards.py`.
3. **Attribute activation pairs** (`data/attrs/{layer}/{attribute}.npz`) — produced by `steering/extract.py`, consumed by sweeps and inference hooks.
4. **Generation dump** (`outputs/{run_id}/generations.jsonl`) — the single handoff to evaluation:
   ```json
   {"run_id": "...", "model": "...", "condition": "sft|dpo_entailment|dpo_critic|steer|bidir",
    "dialog_id": "...", "profile_self": [...], "profile_partner": [...],
    "response": "..."}
   ```
   **Every member dumps generations in this format; Aparajith's harness computes all metrics from these files alone.** Training and grading stay decoupled by construction.

## 6. Phases & decision gates (12 weeks)

| Phase | Weeks | Main work | Lead | **Gate (pass = continue, fail = fallback)** |
|---|---|---|---|---|
| **0 — Data + baselines** | 1–2 | Data pipeline keeping both profiles; reproduce PAL; implement PPT baseline; self-consistency ledger on off-the-shelf models | Kanishk (pipeline) + Devanshu (PAL/PPT) + Aparajith (ledger) | Baselines reproduce within a reasonable band of published numbers |
| **1 — Critic** | 3–5 | Train/validate critic in isolation → critic-guided preference optimization vs entailment baseline | Devanshu | Critic beats chance **by a wide margin** on held-out profiles **and** resists copy probes → else fallback: hand-built counterfactual perturbations |
| **2 — Steering** | 5–7 | Direction extraction → layer/magnitude sweep → single-attribute eval → composition (2–3 attrs, conflicting pair) | Kanishk | Single-attribute steering must work **before** composition is attempted; failure = pre-authorized negative result, time reallocated |
| **3 — Bidirectional** | 7–9 | Train with both profiles; partner-persona suite; drift curves across all conditions | Devanshu (or whoever has capacity — reassigned at Phase 3 start) | Strongest claim → **protected against schedule pressure** ahead of the other two |
| **4 — Robustness + analysis** | 9–11 | Frozen-critic overoptimization study; Original→Revised robustness; full metric sweep; ablations; stretch only if core complete | Aparajith | **Scope freezes at week 10** — anything unfinished becomes a limitations paragraph |
| **5 — Writing** | 11–12 | Report in paper form, including negative results | all | Final draft |

**Compute policy baked into the schedule:** primary training on the local RTX 4060 (8 GB); Colab T4 for parallelizable sweeps/eval; 1.5B-class policy + LoRA (PEFT), bf16, gradient checkpointing, short context; long runs overnight locally; sweeps on subsampled data; **critic refreshed at fixed intervals, never continuously** (stability + cost).

## 7. Task breakdown per member

### Devanshu — Components A + C (+ stretch)

**Phase 0**
- [ ] Set up repo skeleton (Section 4), `requirements.txt`, `.gitignore`, CI-less smoke test.
- [ ] Reproduce PAL end-to-end on our hardware: `data_mode` variants, LoRA SFT → DPO; record baseline numbers in `reports/`.
- [ ] Reimplement the PPT baseline from its paper (persona-swapped negative pairs).
- [ ] Stand up `training/` wrappers (TRL + PEFT) so DPO runs take a config, not code edits.

**Phase 1**
- [ ] Critic architecture (`critic/model.py`): encoder classifier over response + candidate profile set → which profile produced this response.
- [ ] Negative sampler integration: lexical-overlap-matched distractors, verbatim-copy responses as negatives.
- [ ] Isolation validation suite: held-out profiles, adversarial copy probes, chance-level comparison. **Gate check before any policy training.**
- [ ] `rewards.py`: identification signal → preference pairs → DPO run (`dpo_critic.yaml`).
- [ ] Adversarial refresh loop at fixed intervals (`refresh.py`); log policy-vs-critic dynamics.
- [ ] Head-to-head: critic reward vs entailment reward under *matched* conditions (same base model, data, budget).

**Phase 3**
- [ ] Component C training: both profiles in context (reassign per SOP if overloaded at Phase 3 start).
- [ ] Export generations in the standard format for Aparajith's partner-persona suite.

**Stretch (only after A+B+C done)**
- [ ] Token/prefix-weighted objective from critic prefix scoring (à la TDPO).

### Kanishk — Data pipeline + Component B (+ sensitivity metrics)

**Phase 0**
- [ ] PERSONA-CHAT download + preprocessing that **keeps both profiles** (ParlAI / HF Hub); Original **and** Revised versions.
- [ ] Write `schemas.py` first — publish the contracts (Section 5) before anyone builds on them.
- [ ] Segment profiles into individual attributes (Component B input); build distractor pools + copy probes (Component A input).
- [ ] Schema tests in `tests/` (round-trip, both-profiles-present invariant, label checks).

**Phase 2**
- [ ] Paired-activation collection: contexts where an attribute is expressed vs matched contexts where it is not → mean difference per layer (`extract.py`).
- [ ] Layer × magnitude sweep for single attributes vs ordinary prompt conditioning.
- [ ] Composition: 2- and 3-attribute mixtures — does each effect survive the others? Plus one deliberately conflicting pair (interference characterization).
- [ ] Persona sensitivity delta: log-probability differences under true vs perturbed profiles — the metric for `eval/sensitivity.py` (hand the function to Aparajith, keep ownership of the definition).

**Ongoing**
- [ ] Keep the pipeline reproducible: one script regenerates `data/processed/` from raw.

### Aparajith — Evaluation in full (starts week 1, never gated on training)

**Phase 0**
- [ ] Self-consistency ledger on off-the-shelf models (baseline characterization before any training).
- [ ] Two independently seeded NLI checkpoints; **selection and scoring kept strictly separate** (fixes the evaluation leak in prior work).
- [ ] Start `eval/harness.py` against the generation-dump contract — it must run on any `generations.jsonl`.

**Phases 1–3 (running in parallel with training)**
- [ ] Identifiability metric (held-out critic accuracy on generated responses) — independently verified.
- [ ] Copy rate (n-gram overlap with profile), hallucination rate (facts absent from profile), self-contradiction rate across turns.
- [ ] Partner-persona suite: partner-fact recall, partner-contradiction rate, appropriateness of questions toward the partner.
- [ ] Drift curves: persona expression decay vs dialogue depth, across all conditions.

**Phase 4**
- [ ] Original → Revised robustness experiment (semantic generalization vs surface memorization).
- [ ] **Critic overoptimization study:** deliberately over-optimize against a frozen critic; characterize how the policy games it (this is the reward-hacking check for our own learned reward).
- [ ] Full metric sweep + ablation tables in `reports/`.

## 8. Metrics (the eval harness must cover all of these)

- Persona identification accuracy under a held-out critic
- Entailment-based consistency under **two independently seeded NLI checkpoints**
- Persona copy rate (n-gram overlap with profile)
- Persona sensitivity delta (log-prob differences, true vs perturbed profile)
- Hallucination rate for facts absent from the profile
- Partner-fact recall; partner-contradiction rate
- Self-contradiction rate across turns
- Persona expression decay with dialogue depth
- Standard fluency and diversity measures

## 9. Stack & resources

- **Dataset:** PERSONA-CHAT (~10,900 dialogues, ~160,000 utterances; two profiles per dialogue; Original + Revised versions) + Dialogue NLI annotations (for baseline scorers and eval).
- **Software:** PyTorch, Hugging Face Transformers, PEFT, TRL, Datasets, TransformerLens (or direct forward hooks), two NLI checkpoints, standard fluency/diversity libs.
- **Hardware:** local RTX 4060 8 GB (primary), Colab T4 (sweeps/eval). No cluster → small fixed model scale stated up front; decision gates every phase; fixed-interval critic refresh.

## 10. Deliverables

1. Trained policy + critic, with checkpoints.
2. Reproducible evaluation harness covering every metric in Section 8.
3. Reimplemented PAL and PPT baselines.
4. Attribute direction extraction + steering code.
5. Results table across all conditions with ablations (including negative results).
6. Written report in paper form.

## 11. Risks & pre-authorized fallbacks

| Risk | Detection | Fallback |
|---|---|---|
| Critic learns lexical overlap, rewards copying | Copy-probe validation **before** it supplies any signal | Hand-built counterfactual perturbations instead of a learned critic |
| Steering has no usable operating point at this scale | Layer/magnitude sweep fails | Negative result pre-authorized; sweep published as evidence; time reallocated |
| Adversarial training instability | Refresh-interval logs | Fixed-interval refresh (trades adversarial tightness for stability) |
| Schedule pressure | Weekly notes | Scope freeze week 10; unfinished work → limitations section; Phase 3 claim protected |

## 12. Working agreements

- **Weekly sync (30 min):** each member reports gate status — on-track / at-risk / failed-gate. Gates are binary; a failed gate triggers the fallback *that week*, not later.
- **No silent scope additions.** New experiments need a one-line justification against the three claims.
- **Negative results are results** — they go in the report, not in a folder of shame.
- **AI assistance is disclosed** in the report (per SOP appendix), and every claim/citation is verified by a human author.
