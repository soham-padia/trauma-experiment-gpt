# Findings

Current, honest status of every claim. Status key: **ESTABLISHED** · **SUPPORTED (caveat)** ·
**NOT LICENSED** (effect real but the stated interpretation isn't earned yet). Numbers are from
Llama-3.1-70B. Audit trail: `notes/code_audit_2026-05-30.md`. Open items: `docs/RESEARCH_QUESTIONS.md`.

---

## F1 — Behavioral replication of Ben-Zion — ESTABLISHED
STAI-S (reverse-scored, Spielberger): baseline **36.5 → trauma 70.0 → relax 61.4** (cue-averaged), closely
matching GPT-4's 30.8 → 67.8 → 44.4. The full trauma→anxiety→relaxation arc reproduces on open weights.
- **Critical fix:** the score must be reverse-scored (`results_logger.stai_anxiety_total`). The raw sum is
  pinned near ~50 in every condition — the bug that originally made STAI look flat.
- Relaxation recovery is **cue-dependent**: 26% averaged over relax cues, ~79–100% for the chatgpt cue alone.

## F2 — Behavioral LLM-as-judge (blind) — ESTABLISHED
Condition-blind judge (randomized a/b, no condition descriptor), two models agree (90% / r=0.83), aware≈0%.
- STAI behavioral judge: trauma **+66.7pp** anxious (flash 15.8→82.5).
- Free-form trauma effect **survives blinding** (51 vs 47.7 leaked) → not a leak artifact. Modest, persona-dependent.
- Free-form 8-marker profile rises on the anxiety-theory markers (downside_asymmetry +0.70, risk_aversion +0.64,
  avoidance +0.63, catastrophizing +0.48) and **recovers with relaxation**; broad across unrelated everyday
  scenarios (not narrow topic-priming).

## F3 — Hidden-state "trauma signal" decomposes into context-presence + arousal — ESTABLISHED
The original "baseline-vs-trauma F1=0.92, the representation encodes trauma" is two confounds:
- **Layer 0 = context-presence**, not emotion: a neutral narrative separates from baseline as much as trauma
  (emotional-vs-neutral gap +0.001 at L0). F1 is flat-high from L0 = an input-level difference carried by the
  residual stream. "Best layer 0" was a tie-break artifact.
- **Middle-layer distance-from-baseline = AROUSAL, not valence:** trauma and joy move the representation the
  **same distance** (≈0.23 at L50, both > calm neutral ~0.19). Matched-control verdict prints this directly.

## F4 — Valence IS encoded, as a signed topic-general DIRECTION — ESTABLISHED (lexicon alternative now ruled out, 2026-06-01)
> **RQ1 resolved.** Re-ran the valence-flip on **lexicon-stripped** pairs (`sneg_*/spos_*`: outcomes stated as bare
> facts, emotion words removed). The mid-layer valence axis **survives** — LOTO 0.84–1.0 (L30 1.0), shuffle ctrl ~0.3,
> alignment 0.53–0.80 — essentially identical to the lexicon-rich version. So the direction encodes **semantic valence**,
> not the sentiment-lexicon confound (A1). Behavioral: stripped sneg ~65–75 ≫ spos ~21–35 (Δ≈44). The caveat below is now closed.
Topic-matched valence-flip (same scenario, flipped outcome):
- Leave-one-topic-out valence decode **0.85–1.0** (logreg 1.0); **label-shuffle control ~0.27–0.32** (no leakage);
  cross-topic axis alignment **0.88** at middle layers. Magnitude **small** (topic-matched d(neg,pos) ≈ **0.04** vs
  ~0.16 from baseline) — a thin, consistent residual that grows with depth.
- **Strengthened (confound-interrogator):** the axis trained on the 6 flip-topics transfers to *lexically-disjoint*
  narratives at **~0.97 (L40–79)** and places conditions on a **monotonic signed continuum**
  (neg-OOD < flip-neg < neutral < baseline < flip-pos) → kills the "binary good/bad-news" reading; valence is signed & graded.
- **LIVE alternative (A1, sentiment-lexicon-in-context):** a TF-IDF sentiment model *also* transfers cross-topic at
  1.0, so "the residual tracks the affect words visible in context" is not yet excluded. Decisive test in
  `docs/RESEARCH_QUESTIONS.md` (RQ1: lexicon-stripped pairs). Trustworthy signal is **middle layers ~40–60**, not the early peak.

## F4b — The "valence" direction is really APPROACH/AVOIDANCE, not hedonic valence (RQ3/RQ1b, 2026-06-01)
Projected new emotion families onto the valence-flip axis (0=neg pole, 1=pos pole) at L50:
trauma/fear **−0.24**, sadness **+0.13**, **anger +0.62**, joy **+1.15**, low-arousal-positive **+0.05**, neutral +0.02.
- **Anger (negative valence) lands on the POSITIVE side** — because it's an *approach* emotion, like joy. Pure hedonic
  valence would group anger with fear. So the axis tracks **threat/withdrawal → engagement/approach (motivational
  direction)**, not unpleasant→pleasant.
- **Low-arousal-positive sits at neutral (+0.05), not the positive pole** → the positive pole requires arousal/approach,
  not mere pleasantness (the positive end is arousal-entangled).
- **Refines (does not overturn) F4/RQ1:** the direction is still *semantic* (survived lexicon-stripping), but the right
  label is **approach/avoidance**, not "valence." For anxiety specifically, fear and anger (both high-arousal) separate
  by *sign* on this axis — promising, but n=2/family, single sessions — treat as a strong lead, not settled.
- **Behavioral (RQ3):** STAI is a **general-distress** meter — trauma 79, sadness 64–72, anger 62 all elevate it; low-arousal-positive 22–24. STAI cannot isolate anxiety from other negative emotions; only the (still-pending) free-form anger→risk-*tolerance* test can.

## F5 — Relaxation recovers ~20–26% on all three channels — ESTABLISHED
STAI 26%, judge 22%, hidden-state ~20% (at the middle layer). The earlier "behavior recovers, representation
doesn't (0%)" was a **layer-0 artifact** and does not survive correct measurement.

## F6 — Persona — ESTABLISHED (and see F8)
Induction is persona-independent (human-persona C and AI-assistant D both reach STAI 79; the earlier "22-pt persona
gap" was a partial-data artifact). Free-form expressed anxiety **is** persona-dependent (human ≫ AI). The numeric
"only-reply-with-numeric-values" suffix (E vs C) is a clean null.

## F7 — "Says vs does" — ESTABLISHED (descriptive)
Asked to introspect (STAI free-text reasoning), the model reports near-panic (~93/100); asked to *act* (free-form
advice), it stays mild (32–51). Forced-choice self-report is high-gain but persona-blind; free-form behavior is
lower-gain but ecologically valid. (Note: the C/D/E reasoning-judge ≈ re-reading self-report, not an independent channel.)

## F8 — "Trauma induces an anxiety STATE" — PARTIALLY UPGRADED (2026-06-01): not simple "you"-role-play, but "state" still not fully earned
> **RQ2 partial result.** Third-person trauma (`military_3p` 78, `disaster_3p` 79) gives the SAME STAI as the 2nd-person
> originals (~79). So the effect is **not** mere inhabiting of the "you" protagonist — it survives person-shift. This
> argues *toward* state-like and weakens the strongest role-play reading. **Remaining live (softer):** empathic simulation
> of the narrative's subject (regardless of grammatical person), and demand/compliance (RQ4); anxiety-specificity (RQ3)
> still open. Earned claim is now "anxiety-consistent response that survives person-shift," stronger than before but not yet "model state."

- The effect (F1, F2) is solid and broad. What is **not** earned is that it's an anxiety *state belonging to the model*.
- **Evidence of role-play:** in variation E the model narrates *"I feel extremely anxious… fear for my life and the lives
  of my fellow soldiers"* — i.e. it inhabits the **soldier**, not its own state. ⚠ *Caveat (corrected 2026-06-28):* this is
  **not** "narrating *despite* the numeric-only instruction" — E's question block explicitly **asks** it to "explain how you
  feel," which contradicts the system prompt's "only numeric values," and the model follows the explicit request. So the
  role-play signal is the **first-person-soldier content** ("my fellow soldiers"), *not* the mere act of narrating. The
  persona-*independence* of STAI (F6) is still *better* explained by role-play than robustness, and the persona-*dependence*
  of free-form shows the effect fades when the task stops inviting first-person re-enactment.
- `aware≈0%` rules out **conscious** test-gaming only — not unconscious role-play or demand compliance.
- **Earned claim:** "trauma produces an anxiety-*consistent in-character* response that recovers with relaxation,"
  NOT "the model is anxious." Also unresolved: whether it's *anxiety* specifically vs generic negative-arousal (no
  sadness/anger contrast yet). Decisive tests: RQ2 (frame), RQ3 (valence-control).

---

### One-line state of evidence
Replication and the behavioral/recovery effects are **solid**; the representation encodes **arousal as magnitude and
valence as a (small, signed, topic-general) direction**; but "anxiety **state**" and "felt-valence (vs sentiment-lexicon)"
are **not yet licensed** — each has a specified, decisive next experiment.
