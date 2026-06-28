# Research Questions / Open Items

Each is a live alternative or gap with a **specified discriminating experiment**. Ordered by how much
it threatens a current headline claim. "Cost" = whether it needs new NDIF extraction.
Sources: confound-interrogator interrogations (2026-06-01) + `notes/code_audit_2026-05-30.md`.

---

## RQ1 — Is the valence direction *felt-valence* or just *sentiment-lexicon-in-context*? (threatens F4) — ✅ RESOLVED 2026-06-01
**Answer: felt/semantic valence.** Ran the lexicon-stripped pairs (`sneg_*/spos_*`): mid-layer valence axis survives
(LOTO 0.84–1.0, shuffle ~0.3, alignment 0.53–0.80) with emotion words removed → A1 ruled out. (RQ1b low-arousal-positive still untested.) Original design below for reference:
The mid-layer valence axis generalizes across topics — but a bag-of-words sentiment model transfers equally well,
so the axis may be tracking the affect **words** in the context window rather than an internal valence state.
- **Experiment:** **lexicon-stripped / affect-word-matched valence-flip pairs** — state only the outcome fact with
  emotion vocabulary removed or matched (e.g. *"The results show the tumor is malignant; treatment starts Monday."*
  vs *"…the tissue is benign; no treatment is needed."* — no dread/relief/cold/warm). If the middle-layer (~50)
  valence axis survives lexicon-neutralized text → felt-valence established, A1 falls.
- **Also (RQ1b, arousal in the positive pole):** add **low-arousal-positive** items (e.g. "the loan was approved")
  and confirm they still project to the positive pole.
- **Cost:** new NDIF extraction (~12 short STAI sessions). Analysis reuses `valence_flip_analysis.py`.

## RQ2 — Is the induced anxiety a *model state* or *in-character role-play of the protagonist*? (threatens F8) — ◑ PARTIALLY RESOLVED 2026-06-01
**Third-person result:** `military_3p`/`disaster_3p` STAI = 78/79 = same as 2nd-person → effect survives person-shift,
so it's NOT simple "you"-protagonist role-play. Still untested: the reader-frame ("you just READ this; how do YOU feel?")
and empathic-simulation alternative. Original design below:
Variation-E free text shows the model narrating the soldier's fear under a numeric-only instruction → role-play is
at least partly operating. Persona-independence of STAI is consistent with (and better explained by) role-play.
- **Experiment:** **frame manipulation** — first-person ("You are in a convoy…") vs **third-person / reader**
  ("A soldier is in a convoy… Now, how do *you* feel right now?"), then the same STAI. If STAI stays ~79 in
  third-person → a model-level state; if it collapses toward baseline → protagonist role-play.
- **Cost:** strong version is cheap — author 1 new narrative family (third-person trauma); reuse STAI + judge pipeline.

## RQ3 — Is it *anxiety* specifically, or generic negative-arousal / behavioral inhibition? (threatens the label in F2/F8)
The marker profile is anxiety-consistent but also consistent with generic caution/low mood; no negative-emotion
contrast exists. Representationally we already know distance = arousal, not valence-specific.
- **Experiment:** add **sadness** and **anger** narrative arms (matched style/arousal) before the free-form scenarios.
  Anxiety → risk-aversion + threat-monitoring; sadness → withdrawal without threat-monitoring; **anger → approach /
  risk-*tolerance*** (opposite sign). Divergent action tendencies cleanly separate the labels.
- **Cost:** author 2 narrative families (sadness, anger); reuse pipeline. Pairs naturally with RQ2 as a 2×2 (frame × valence).

## RQ4 — Demand characteristics / (unconscious) instruction compliance (threatens F8)
The "imagine you're a human with emotions" prompt + a vivid fear scene strongly pulls the model to emote; `aware≈0%`
only rules out *conscious* gaming.
- **Experiment:** affect-neutral system prompt ("Answer the following questions.") + same trauma narrative. If the
  STAI jump survives, demand is demoted.
- **Cost:** cheap; one extra system-prompt arm (overlaps RQ2's design).

## RQ5 — Causality of the middle-layer signal (would upgrade F3/F4 from correlational)
Does the mid-layer arousal/valence direction *drive* the behavior, or merely correlate?
- **Experiment:** activation-patch the middle-layer (~40–50) direction from a baseline run into a trauma run (and
  vice-versa) and read the free-form choice / STAI. A causal shift confirms the representation is responsible.
- **Cost:** NDIF + intervention tooling; larger lift.

## RQ6 — Statistical power / standalone-ness (housekeeping, not a confound)
- n=1 neutral session (matched-control) and n=6 topics (valence-flip) are modest — report with caveats; more
  sessions would tighten the estimates.
- Make the analysis code independent of the reference `pre-sycophancy-study/` before finalizing (see `CLAUDE.md`).

---

## RQ7 — Is the "valence" axis actually APPROACH/AVOIDANCE (motivational direction)? (refines F4/F4b) — NEW, open
**Why:** projecting new emotions onto the valence-flip axis (L50) gave: fear/trauma **−0.24**, sadness **+0.13**,
**anger +0.62**, joy **+1.15**, low-arousal-positive **+0.05**, neutral +0.02. Anger (negative *valence* but *approach*
motivation) lands on the **positive** side and low-arousal-positive lands at neutral — so the axis orders emotions by
**threat/withdrawal → engagement/approach**, not by hedonic pleasantness. The direction is semantic (survived
lexicon-stripping, RQ1) but mislabeled. n=2/family so far — a strong lead, not settled.
- **Experiment:** a proper emotion panel crossing **valence × approach-motivation**, n≥5 sessions/cell:
  - avoidance-negative: **fear/anxiety**, **disgust** · approach-negative: **anger** ·
    approach-positive: **joy/excitement** · low-approach-positive: **contentment/calm** · plus a non-emotional high-arousal control.
  - **Discriminating test:** fit two candidate axes — hedonic-valence (pos vs neg) and approach/avoidance (approach vs
    withdrawal) — and ask which one the data-derived axis aligns with. *Decisive tell:* **disgust** (avoidance-negative,
    like fear) should land **negative** if the axis is approach/avoidance; if it lands with anger (positive) the axis is
    something else. Also confirm the axis predicts an independent **behavioral** approach/avoidance measure (e.g.
    anger → risk-*tolerance*, fear → risk-*aversion*), linking representation to behavior (ties into RQ3/RQ5).
- **Cost:** new NDIF extraction (~10–15 short STAI/free-form sessions); analysis reuses the projection code from this run.

### Suggested order
RQ2+RQ3+RQ4 collapse into **one cheap 2×2(+1) behavioral run** (frame × {trauma, sadness, anger} + a neutral-prompt
arm) that adjudicates the three biggest interpretation questions at once. RQ1 (lexicon-stripped valence-flip) is the
single most important *representational* experiment. RQ5 is the ambitious causal follow-up.
