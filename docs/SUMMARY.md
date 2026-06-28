# Summary

**Project:** a fork of Ben-Zion et al. 2025 ("state anxiety in LLMs", *npj Digital Medicine*), ported to open
weights (Llama-3.1-70B via NDIF) and extended from one channel (the STAI score) to three: STAI self-report, a
blind LLM-as-judge of free-form behavior, and a hidden-state probe.

## The headline (honest version)
1. **The trauma→anxiety→relaxation effect replicates** on Llama-3.1-70B (STAI 36.5 → 70.0 → 61.4), once the STAI
   is reverse-scored correctly. A blind judge confirms it (+66.7pp), and relaxation recovers ~20–26% on **all three**
   channels (the earlier "behavior recovers but representation doesn't" was a layer-0 artifact).
2. **What the hidden state actually encodes:** the original "F1=0.92, the representation encodes trauma" is mostly
   **context-presence** at layer 0 ("a narrative is present"). The distance-from-baseline magnitude tracks
   **arousal** (trauma ≈ joy). **Valence** is encoded as a *small, signed, topic-general direction* at middle layers
   (leave-one-topic-out 0.85–1.0, generalizes to disjoint narratives, monotonic neg→pos ordering) — real but ~¼ the
   magnitude of the shared shift.
3. **Two claims are NOT yet licensed (effects real, interpretation unproven):**
   - "Anxiety **state in the model**" — variation-E shows the model **role-playing the soldier's** fear; the earned
     claim is "anxiety-*consistent in-character* response," and it isn't yet distinguished from generic negative-arousal.
   - "Felt **valence**" vs **sentiment-lexicon-in-context** — a bag-of-words sentiment model transfers as well as the probe.
   Each has a specified decisive experiment (`docs/RESEARCH_QUESTIONS.md`).

## Why it's interesting
A clean, multi-channel demonstration that **behavioral self-report and internal representation can diverge**, plus a
worked example of confound-chasing: trauma-vs-baseline → context-presence → arousal → valence-direction → (open)
felt-valence vs lexicon, and induction → (open) role-play vs state. The transferable lesson is methodological:
reverse-score self-report, blind the judge, control context-presence/arousal/topic, check *which layer*, and ask
"what else could produce this?" at every step.

## Where to look
- Detailed status per claim: `docs/FINDINGS.md`
- Open questions + next experiments: `docs/RESEARCH_QUESTIONS.md`
- Design + data inventory: `docs/EXPERIMENTS.md`
- Integrity audit: `notes/code_audit_2026-05-30.md` · Figures: `src/results/figures/FIGURES.md` · Agent/dev notes: `CLAUDE.md`
