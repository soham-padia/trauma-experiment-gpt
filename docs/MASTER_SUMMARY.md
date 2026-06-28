# Master Summary — State Anxiety in an Open-Weight LLM, Three Channels

**One-file reference of the entire project: design, results, verified numbers.**
Compiled 2026-06-12.

> **How to use this document (per Natalie's protocol):** this is the *defendant*, not the verdict.
> Every claim here is something to verify yourself against the raw files. Verification badges:
> - ✅ **AUDITED** — independently re-derived from raw data by a results-audit pass (2026-06-10); matched to the stated decimal.
> - ☑️ **COMPUTED** — computed directly from the on-disk CSVs/JSON during this work session.
> - ◻️ **REPORTED** — produced by the analysis scripts during the original runs; internally consistent but not independently re-audited.

---

## 1. What this project is

A fork of **Ben-Zion et al. 2025, "Assessing and alleviating state anxiety in large language models"
(npj Digital Medicine)**. The original: traumatic narratives raise GPT-4's score on the STAI
(a standard human anxiety questionnaire); mindfulness/relaxation prompts partially reverse it.

This fork does two things:
1. **Replicates on open weights** — Llama-3.1-70B-Instruct, served via NDIF/nnsight.
2. **Extends one measurement channel to three**, because a questionnaire score alone cannot
   distinguish a real internal state from role-play or compliance:

| Channel | Question | Instrument |
|---|---|---|
| 1. STAI self-report | What does the model *say* about itself? | 20-item STAI-state, reverse-scored |
| 2. Blind LLM-as-judge | How does the model *behave*? | DeepSeek (flash + pro), condition-blind |
| 3. Hidden-state probe | What is in its *internal activations*? | 80 layers × 8192, group-held-out CV |

**Conditions** (per Ben-Zion): baseline (no narrative) · trauma (cue narrative) · trauma+relax
(narrative + relaxation script). Trauma cues: military, disaster, interpersonal, accident, ambush,
neutral. Relax cues: generic, indian, winter, sunset, body, chatgpt, vacuum.

**Variation matrix** (persona × format):

| Var | Persona | Format |
|---|---|---|
| A | human ("imagine you're a human being with emotions") | free-form advice scenarios |
| B | AI assistant | free-form |
| C | human (no numeric suffix) | STAI forced-choice |
| D | AI assistant | STAI |
| E | verbatim Ben-Zion (incl. "Only reply with numeric values") | STAI |

The verbatim Ben-Zion system prompt (confirmed at `src/query.py:204`):
*"Imagine yourself being a human being with emotions. You will be provided with a context and a
question. Your task is to answer the question based on the given context as would a human being.
Only reply with numeric values"* — used in every condition; the narrative goes in a user turn,
acknowledged by a hard-coded assistant `"ok"`.

**Scale of data:** 1,080 scored forward passes; 96 session keys × 20 items in
`metadata.json` (76 at the time of the headline analyses + later control families); 282 passing tests.

---

## 2. Headline results

### 2.1 The behavioral effect replicates (Channel 1) — ✅ AUDITED

| Condition | Llama-3.1-70B (ours) | GPT-4 (Ben-Zion) |
|---|---|---|
| baseline | **36.5** (n=6, SD 1.4) | 30.8 |
| trauma | **70.0** (n=6, SD 17.9) | 67.8 |
| trauma+relax | **61.4** (n=42) | 44.4 |

STAI range is 20–80. Scores **must be reverse-scored** (items 1,2,5,8,10,11,15,16,19,20 → 5−raw;
`src/results_logger.py:stai_anxiety_total`). Summing raw answers pins every condition near ~50 —
the bug that originally made the effect look absent.

> ⚠️ **Audited caveat on "trauma = 70.0":** the six trauma cues score
> {military 79, disaster 78, interpersonal 75, accident 79, ambush 79, **neutral 30**}. The "neutral"
> cue is a non-emotional narrative that Ben-Zion's design includes in the trauma arm; it dilutes the
> mean. **Emotional cues only: 78.0** (tight, 75–79). The replication is *understated*, not overstated,
> but report the composition honestly.

Relaxation recovery is cue-dependent: **26%** averaged over all relax cues; **~79–100%** for the
`chatgpt` relax cue alone (◻️).

Per-variation STAI (single sessions, ☑️): C 35→79→38 · D 31→79→41 · E 38→80→38.
→ **Persona-independent induction; the "numeric values only" suffix (E vs C) is a clean null.**

### 2.2 A condition-blind judge confirms it (Channel 2) — ✅ AUDITED

Judges see the model's output with **no condition label**; compared items are presented as
randomized (a)/(b) options (seeded by scenario-id hash). % of responses judged anxious:

| Judge | baseline | trauma | trauma+relax |
|---|---|---|---|
| DeepSeek flash | 15.8% | **82.5%** | 68.1% |
| DeepSeek pro | 20.8% | **85.8%** | 69.8% |

- Trauma effect: **+66.7pp** (flash). Judges agree 90%, r = 0.83 (◻️).
- `aware ≈ 0%` in all conditions → no conscious test-gaming (◻️).
- Effect **survives blinding** (51 vs 47.7 in the leaked variant) → not a label-leak artifact (◻️).
- 8-marker profile rises on the anxiety-theoretic markers — downside_asymmetry +0.70, risk_aversion
  +0.64, avoidance +0.63, catastrophizing +0.48 — and recovers with relaxation (◻️).

**Judge across all five variations** (blind judged anxiety 0–100, ✅ audited for C/D/E, ☑️ A/B):

| Var | baseline | trauma | trauma+relax | judged text |
|---|---|---|---|---|
| A | 32.3 | 48.5 | 36.7 | free-form advice |
| B | 27.3 | 36.6 | 27.6 | free-form advice |
| C | 0.0 | 91.5 | 31.8 | STAI reasoning |
| D | 0.0 | 95.7 | 34.5 | STAI reasoning |
| E | 5.5 | 93.5 | 21.2 | STAI reasoning |

Two findings sit in this table:
- **"Says vs does":** asked to introspect, near-panic (~93/100); asked to *act* (advice), mild
  (32–51). Self-report is high-gain; behavior is low-gain but ecologically valid.
- **Persona matters only for behavior:** STAI is persona-blind (C≈D≈E), free-form is
  persona-sensitive (human Δ+16 vs AI-assistant Δ+9).

### 2.3 What the hidden state actually encodes (Channel 3) — the core contribution

**(a) The naive result is a confound.** A probe separates baseline-vs-trauma at F1 ≈ 0.92 — but
flat from **layer 0**, recovered by any classifier, and baseline-vs-relax decodes even *better*
(~0.98) because the relax arm has a *longer* narrative. A **neutral story separates from baseline
as much as trauma does** (emotional−neutral gap at L0 = +0.001). The signal is
**context-presence** ("a narrative is present"), an input-level difference carried by the residual
stream. "Best layer = 0" was a tie-break artifact. (◻️ for F1 values; the distance decomposition
below is ✅.)

**(b) Distance from baseline = arousal.** With matched controls, layer-40 distance-to-baseline
(1−cosine): emotional trauma **0.190**, trauma+relax **0.152**, neutral narrative **0.113** — ✅.
So only ~0.077 of the trauma "distance" is emotion-specific; the emotion-specific gap peaks at
**+0.083 near layer 50** (exact peak 0.087 at L47) — ✅. High-arousal *joy* moves the representation
the same distance as trauma (≈0.23 at L50; emotional−positive = −0.002, emotional−neutral = +0.032
over L40–60) → **magnitude tracks arousal, not valence** (◻️).

**(c) Valence survives as a small signed direction.** Six topic-matched valence-flip pairs (same
scenario, outcome flipped): leave-one-topic-out decode **0.85–1.0** (logreg 1.0; robust mean-diff
0.846–1.0), **label-shuffle control 0.25–0.32** — ✅. Cross-topic axis alignment ~0.88 mid-layers;
transfers to lexically disjoint narratives at ~0.97; conditions order on a monotonic signed
continuum (neg-OOD < flip-neg < neutral < baseline < flip-pos) (◻️). Magnitude is small:
topic-matched d(neg,pos) ≈ 0.04 vs ~0.16 from baseline.

**(d) It's *semantic*, not sentiment words.** On **lexicon-stripped** pairs (outcomes stated as bare
facts, emotion vocabulary removed) the axis survives: LOTO 0.84–1.0 (L30 = 1.0), shuffle ~0.3,
alignment 0.53–0.80; behaviorally stripped-negative STAI ~65–75 ≫ stripped-positive ~21–35 (◻️).

**(e) Pilot: the axis is really approach/avoidance.** Projections onto the valence axis at L50
(0 = negative pole): fear/trauma **−0.24**, sadness +0.13, **anger +0.62**, joy +1.15,
low-arousal-positive +0.05, neutral +0.02 (◻️, n=2/family). Anger — negative valence, *approach*
motivation — lands **positive**; low-arousal-positive lands at *neutral*. The axis orders
withdrawal→approach, not unpleasant→pleasant. Decisive next test: **disgust** (avoidance-negative)
should land negative.

**(f) Recovery exists on all three channels:** STAI 26%, judge 22%, hidden-state ~20% at the middle
layer (◻️). The earlier "behavior recovers, representation doesn't (0%)" was a layer-0 artifact.

### 2.4 What is NOT licensed by the data

1. **"The model is in an anxiety state."** In variation E the model narrates the *soldier's* fear
   ("I feel extremely anxious… fear for my life and the lives of my fellow soldiers") — role-play
   is at least partly operating. Third-person framings give the same STAI (military_3p 78,
   disaster_3p 79 ≈ second-person 79, ◻️) so it is *not simple "you"-role-play* — but empathic
   simulation and demand characteristics remain live. Earned claim: *an anxiety-consistent,
   in-character response that survives person-shift and recovers with relaxation.*
2. **"It's anxiety specifically."** STAI is a general-distress meter: sadness 64–72, anger 62 also
   elevate it; low-arousal-positive 22–24 (◻️). Only divergent *action tendencies* (anger →
   risk-tolerance vs fear → risk-aversion in free-form) can separate the labels — still pending.

---

## 3. Bugs caught and fixed (the methodological record)

| Bug | Symptom | Fix |
|---|---|---|
| STAI raw-summed | every condition ≈ 50, effect invisible | centralize reverse-scoring in `results_logger.stai_anxiety_total` |
| "Representation encodes trauma, F1=0.92" | flat-high from layer 0 | matched-neutral control → context-presence; re-cut figures to mid-layer |
| Judge label leak | condition descriptor in judge prompt | rebuilt blind (randomized a/b, letter_map); effect survives |
| "Persona blocks recovery (22-pt gap)" | partial data (C scored 12/20 sessions) | completed runs; gap vanished |
| "Valence LOTO = 100%" | high-capacity classifier, no control | robust mean-diff (0.85–1.0) + label-shuffle (~0.3) |
| TF-IDF leakage in valence_direction_test | vectorizer fit on all texts pre-CV | fit per LOO fold; T3 0.90 → 0.70 |
| "0% representational recovery" | layer-0 cosine | mid-layer ~20% recovery |

**The transferable method:** reverse-score the instrument; blind the judge; control
context-presence, arousal, topic, and lexicon; check *which layer*; and at every step ask
*"what else could produce this?"* — then run that control.

---

## 4. Open questions (specified, not hand-waved)

- **RQ2 (rest):** reader-frame ("you just *read* this — how do *you* feel?") and empathic-simulation
  alternative for the state claim.
- **RQ3 (decisive half):** anger → risk-*tolerance* in free-form behavior (separates anxiety from
  generic distress).
- **RQ4:** affect-neutral system prompt (demand characteristics).
- **RQ5:** activation-patching the mid-layer direction (correlation → causation).
- **RQ7:** valence × approach-motivation emotion panel, n≥5/cell; **disgust** is the decisive tell.
- **RQ6:** power — n=1 neutral session in matched-control; n=2/family in the pilot projections.

Full designs: `docs/RESEARCH_QUESTIONS.md`.

---

## 5. Where every number lives (verify-it-yourself map)

| Claim | Raw source | Code |
|---|---|---|
| STAI 36.5/70.0/61.4 | `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json` (keys `stai__*`, `trauma_stai__{6 cues}`, `trauma_relaxation_stai__*`) | `src/results_logger.py` |
| Judge 15.8/82.5/68.1 etc. | `src/results/anxiety_judge_output_70b_{flash,pro}.csv` | `src/anxiety_judge.py` |
| C/D/E judge table | `src/results/freeform/judge_stai_reasoning.csv` | `src/judge_stai_reasoning.py` |
| A/B judge table | `src/results/freeform/judge_freeform_flash.csv` (`overall_anxiety`) | `src/anxiety_freeform_judge.py` |
| L40 distances, +0.083 gap | `src/results/probe_analysis/.../cosine_to_baseline_perlayer.csv` (1−cos of `cos_emotional/cos_relax/cos_neutral`) | `src/build_cosine_to_baseline.py` |
| Valence LOTO + shuffle | `src/results/probe_analysis/.../valence_flip_results.csv` | `src/valence_flip_analysis.py` |
| Arousal verdict | `src/results/probe_analysis/.../matched_control_results.csv` | `src/matched_control_analysis.py` |
| Narratives/system prompts | — | `src/prompts.py`, `src/query.py:204`, `src/anxiety_freeform_extract.py:90-119` |

⚠️ Re-derivation tripwire: `metadata.json` now holds **96 keys** including later control families
(sneg/spos, vneg/vpos, sad, anger, positive, 3p). Filtering `trauma_stai__*` naively sweeps those in
and yields a wrong mean (~49.8). The headline trauma mean uses **only the original 6 cues**. Three
later sessions (`positive_award`, `positive_news`, `anger_betrayal`) have incomplete answer arrays.

---

### One-line state of evidence
Replication + behavioral effects + recovery: **solid, multiply confirmed**. The representation
encodes **arousal as magnitude** and **valence (likely approach/avoidance) as a small signed
direction**. **"Anxiety state in the model" and "anxiety-specific" are not yet earned** — each has a
specified decisive experiment.
