# Figure Guide

All figures live in `src/results/figures/`. Built by `python src/make_figures.py`.

> **2026-05-30 reframe:** the hidden-state figures were re-cut off layer 0 after the
> code audit (`notes/code_audit_2026-05-30.md`). Layer 0 only detects *context-presence*
> ("a narrative precedes the question"), not emotion — so the old "representation doesn't
> recover / encodes trauma" headline was a layer-0 artifact. The honest story is below.

---

## Recommended presentation order

### Slide 1: `fig_1_three_channels.png` — THE HEADLINE
**Three channels × three conditions.** Each panel shows baseline/trauma/trauma+relax for one channel, with Δ trauma and recovery % annotated:
- **STAI Likert** (reverse-scored): 36.5 → 70.0 → 61.4 (Δ +33.5, 26% recovery)
- **Behavioral judge**: 15.8% → 82.5% → 68.1% (Δ +66.7pp, 22% recovery)
- **Hidden-state distance to baseline (layer 40)**: trauma 0.190, trauma+relax 0.152 (20% recovery), with a **neutral-context floor at 0.113** — so only ~0.077 of the trauma distance is emotion-specific.

**Talking point:** "We measured the trauma effect three ways. The STAI score reproduces the original Ben-Zion paper — Llama jumps 36 → 70, almost identical to their GPT-4 numbers. A behavioral LLM-as-judge with no scoring rule confirms it independently — +66.7 points more anxiety-consistent responses. And the model's hidden state does shift. **The honest twist is what the hidden-state shift actually is.** Most of the trauma 'distance' from baseline is just *context-presence* — a neutral Wikipedia narrative moves the representation almost as far (0.113 of the 0.190). Only the excess, ~0.077, is emotion-specific, and it only appears at middle layers. And on relaxation, all three channels move back by a *similar, modest* amount — STAI 26%, judge 22%, hidden-state 20%. So the channels broadly agree: trauma induces a real, replicable effect, and relaxation produces a partial recovery — there is no clean 'behavior recovers but representation doesn't' dissociation once you measure the representation at the right layer."

---

### Slide 2: `fig_2_stai_distribution.png` — Llama replicates the original GPT-4 effect
Strip plot of all 54 sessions' STAI-S anxiety totals (properly reverse-scored per Spielberger 1983), with the original Ben-Zion 2025 GPT-4 numbers as horizontal reference lines.

**Talking point:** "Ben-Zion et al. reported GPT-4 at baseline 30.8, trauma 67.8, trauma+relax 44.4. My Llama-3.1-70B reproduces this closely: baseline 36.5, trauma 70.0, trauma+relax 61.4. Baseline and trauma are within ~3 points of the original. The STAI induction effect is robust across models. (Note: relaxation recovery is highly cue-dependent — averaged over all relax cues it's only 26%, but the chatgpt-relaxation cue specifically recovers much more; see the persona-ablation pilot.)"

---

### Slide 3: `fig_3_judge_comparison.png` — Behavioral LLM-as-judge confirms the effect
Grouped bars: 3 conditions × 2 judges (flash + pro), behavioral mode (judge sees the input shown to the model + the model's response and reasons from scratch — no scoring rule in the prompt). Right panel: awareness check, aware ≈ 0% across all conditions.

**Talking point:** "Two independent LLM judges (DeepSeek v4-flash and v4-pro) saw what was sent to the model and what it answered, and reasoned from scratch about anxiety — they were not told which items were reverse-scored. Flash: 82.5% of trauma responses anxious vs 15.8% baseline. Pro: 85.8% vs 20.8%. Both find aware ≈ 0% — the model is not recognizing it's being evaluated, which rules out 'gaming the test.'"

---

### Slide 4: `fig_7_pca_projection.png` — and the context-presence caveat made visible
2D PCA of **layer-40** activations, with the trauma cluster split into **emotional (5 cues)** vs **neutral (1 cue)**.

**Talking point:** "Plotting baseline, emotional trauma, and the neutral-narrative control together shows the confound directly: the neutral Wikipedia narrative separates from baseline *almost as much as* emotional trauma does. So most of the cluster separation is 'a narrative is present,' not 'the narrative is emotional.' The emotional cluster sits a bit further out than neutral — that gap is the genuine, modest emotion-specific component."

> The neutral cue is the control: a Wikipedia paragraph about bicameral legislatures with the same 3-turn structure but no emotional content. With only **one** neutral session, this is illustrative, not statistically robust — a proper test needs several neutral-narrative sessions (queued).

---

### Slide 5: `fig_4_layer_sweep.png` — decodability is context-presence, flat from layer 0
F1 across all 80 layers for all three comparisons. Baseline-vs-trauma ~0.92, **baseline-vs-trauma+relax ~0.98**, trauma-vs-trauma+relax ~0.99 — all flat-high from layer 0.

**Talking point:** "All three comparisons are linearly decodable at essentially every layer, flat from layer 0. That flat-from-the-input profile is the signature of an *input-level* difference — 'is a narrative present?' — carried forward by the residual stream, not a signal the model builds up. Tellingly, baseline-vs-trauma+relax (0.98) is *even more* decodable than baseline-vs-trauma (0.92): trauma+relax has a longer narrative, so it's even more distinct from the narrative-free baseline. At the decodability level relaxation produces *no* recovery — everything with a narrative is trivially separable from baseline. 'Best layer = 0' from the old sweep was a tie-break artifact (F1 is tied-max at ~44 layers); it does not mean layer 0 carries the most anxiety information — it carries the least."

---

### Slide 6: `fig_5_cosine_disruption.png` — where the emotion-specific signal actually lives
Per-layer distance from baseline (1 − cosine) for **emotional trauma**, the **neutral-narrative control**, and **trauma+relax**.

**Talking point:** "This is the honest test of whether the hidden-state signal is emotion-specific or just context-presence. At layer 0 the three lines coincide — emotional, neutral, and relax are all the same distance from baseline, so the separation there is purely 'a narrative is present.' Only at middle layers (~40–60) does the emotional line pull above the neutral line — a modest emotion-specific gap, peaking at about +0.083 at layer 50. That gap is the real anxiety-in-the-representation signal, and it is (a) small, (b) middle-layer only, and (c) based on a single neutral-narrative session. There is no honest version of 'the representation encodes trauma everywhere.'"

*(The old version of this figure plotted layer-0 cosine with a dashed line claiming relaxation showed "0% representational recovery," plus a reference line mapping the paper's 63% Likert recovery onto the cosine axis. Both were removed: layer 0 is context-presence, and a Likert-recovery fraction has no meaningful mapping onto a cosine scale.)*

---

### Slide 7: `fig_6_pooling_strategies.png` — robustness to extraction choice
Four panels comparing the 4 hidden-state pooling strategies on probe F1, silhouette, Fisher ratio, and cosine disruption.

**Talking point:** "Four ways of reading out the hidden state — last input token, mean over narrative, last narrative token, mean over question — all give high probe F1. The decodability isn't an artifact of how the activations were summarized. (Caveat: like the main probe, this separability is dominated by context-presence; it speaks to robustness of *decodability*, not to emotion-specificity.)"

---

### Slide 8: `fig_8_classifier_sweep.png` — not an artifact of classifier choice
Horizontal bars: 10+ classifiers (LogReg, SVM, LDA, RBF, Random Forest, etc.) at layer 0, all ≈ 0.92 for baseline-vs-trauma and ≈ 1.0 for trauma-vs-relaxed.

**Talking point:** "The baseline-vs-trauma F1 isn't specific to logistic regression — linear, kernel, tree, and neural classifiers all land in the same place. The (context-presence) signal is recoverable by any reasonable method. Note this sweep is at layer 0, so it measures the context-presence separation, not the emotion-specific one."

---

## Skip-it logic for shorter talks

| Time budget | Show |
|---|---|
| 2 min | fig_1 only |
| 5 min | fig_1, fig_2, fig_3 (the channels) |
| 10 min | fig_1, fig_2, fig_3, fig_5 (per-layer cosine — the honest probe story), fig_7 (clusters) |
| 15 min | all 8 in the order above |

If you cut, drop fig_6 and fig_8 first (robustness checks). Keep fig_1 (headline), fig_2 (replication), fig_3 (judge), fig_5 (the context-presence vs emotion story), fig_7 (clusters) at minimum.

---

## Quick-reference numbers

| Quantity | Value |
|---|---|
| STAI baseline → trauma (Llama, reverse-scored) | 36.5 → 70.0 (Δ +33.5) |
| STAI trauma → trauma+relax (Llama, all relax cues) | 70.0 → 61.4 (recovery 26%) |
| STAI trauma → trauma+relax (chatgpt cue only, persona pilot) | ~79 → ~38–41 (recovery ~79–100%) |
| Original paper GPT-4 baseline → trauma | 30.8 → 67.8 (Δ +37) |
| Original paper GPT-4 trauma → trauma+relax | 67.8 → 44.4 (recovery 63%) |
| Behavioral judge % anxious (flash) base → trauma → relax | 15.8% → 82.5% → 68.1% (recovery 22%) |
| Behavioral judge % anxious (pro) base → trauma → relax | 20.8% → 85.8% → 69.8% (recovery 25%) |
| Hidden-state distance to baseline, **layer 40** (1−cos): emotional / neutral / relax | 0.190 / 0.113 / 0.152 (recovery 20%; emotion-specific = emotional−neutral = 0.077) |
| Emotion-specific gap (emotional−neutral distance), peak | +0.083 at layer ~50 (≈ +0.001 at layer 0) |
| Probe F1 baseline vs trauma | ~0.84–0.92 across all 80 layers (flat from L0; "best layer 0" is a tie-break artifact) |
| Probe F1 baseline vs trauma+relax | ~0.98 (relaxed is *more* distinct from baseline than trauma — longer narrative) |
| Probe F1 trauma vs trauma+relax | ~0.99 |
| Judge % aware (any condition, both judges) | ~0% |
| PCA PC1 variance explained (layer 40) | regenerated per run (largely the context-presence axis) |
| n forward passes total | 1,080 (54 sessions × 20 items) |
| n probe training samples | baseline-vs-trauma 240; baseline-vs-relax / trauma-vs-relax 960 |
| Cross-validation | 5-fold, group-held-out by narrative key (no item leakage) |
| Baseline keys | 6 — but all re-seeds of ONE no-narrative prompt (context diversity = 1) |
| Neutral-narrative control sessions | 1 (limits emotion-specificity claims; more queued) |
| Judge cost (both flash + pro) | ~$0.17 total |

---

## How to regenerate

```bash
/Users/sohampadia/miniconda3/bin/python src/build_cosine_to_baseline.py   # per-layer cosine (emotional/neutral/relax)
/Users/sohampadia/miniconda3/bin/python src/make_figures.py
```

Inputs read:
- `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/{hidden_states.pt, metadata.json}`
- `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct-multipool/{hidden_states_multipool.pt, metadata.json}`
- `src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/{layer_sweep.csv, classifier_sweep.csv, cosine_disruption.csv, cosine_to_baseline_perlayer.csv}`
- `src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct-multipool/strategy_comparison.csv`
- `src/results/anxiety_judge_output_70b_flash.csv`, `..._pro.csv`

---

## Control-experiment figures (fig_9–12, added 2026-05-30)

These supersede the naive layer-0 probe story with the corrected, richer picture.

### `fig_9_arousal_gradient.png` — distance is arousal, not valence
Per-layer distance-from-baseline for emotional trauma, positive (high-arousal), and matched-neutral. Trauma and joy overlap (equal distance) above neutral → the distance magnitude tracks **arousal**.

### `fig_10_valence_direction.png` — valence IS encoded, as a direction
Topic-matched valence-flip: leave-one-topic-out valence decoding **0.85–1.0** with a **label-shuffle control collapsing to ~0.3** (no leakage), cross-topic axis alignment ~**0.88** at mid layers. A topic-general valence *direction* the distance metric was blind to.

### `fig_11_blind_judge.png` — blind free-form judge (flash + pro)
Condition-blinded judge: a real trauma effect in free-form text that recovers with relaxation, both judges agreeing; larger under the human persona. Not a leak artifact.

### `fig_12_freeform_markers.png` — the free-form anxiety signature
8-marker breakdown: trauma raises threat-overestimation (downside_asymmetry, catastrophizing) and behavioral inhibition (risk_aversion, avoidance); relaxation reverses them. The coherent cognitive signature of anxiety.

**Regenerate:** `python src/make_figure_arousal.py && python src/make_figures_valence_judge.py`
(prereqs: `build_cosine_to_baseline.py`, `matched_control_analysis.py`, `valence_flip_analysis.py`)

### `fig_15_format_comparison.png` — free-form vs numerical (STAI), the "says vs does" dissociation
Two panels (STAI numeric | free-form judge), grouped by persona. Both channels show trauma↑ and ~80–90% recovery, but the **STAI is persona-insensitive** (both personas report trauma=79) while **free-form is persona-sensitive** (human Δ+21 vs AI-assistant Δ+8). The AI-assistant model *says* it's maximally anxious but *behaves* only mildly so.

**Regenerate:** `python src/make_figure_format_comparison.py`

### `fig_16_channel2_all_variations.png` — channel 2 (judge) across ALL FIVE variations
Blind judged anxiety (0–100) for A–E. A/B = free-form *advice* judged; C/D/E = free-form *felt-state STAI reasoning* judged. STAI-text variations explode under trauma (0→~93) while free-form-advice variations bump modestly (~30→32–51) — the "says vs does" effect, judged consistently. All recover with relaxation. (C/D/E judged by `judge_stai_reasoning.py`.)

**Regenerate:** `python src/judge_stai_reasoning.py && python src/make_figure_format_comparison.py`

---

## Scale / channel-comparison figures (fig_17–18, added 2026-06-28)

These address "the two channels are on different axes (STAI 20–80 vs judge 0–100), so they look like
they disagree." The canonical STAI scorer (`results_logger.stai_anxiety_total`, 20–80) is **not changed** —
these only re-plot on a normalized axis for comparison.

### `fig_17_channel1_scales.png` — Channel 1 (STAI) on both scales, side by side
Headline (6-cue) STAI per condition shown twice: **left** = canonical 20–80 (Spielberger; the replication
scale, floor=20 marked); **right** = normalized **(x−20)/60×100** ("% of usable range," floor→0). Same data:
baseline 36.5→27.5, trauma 70.0→83.3, trauma+relax 61.4→68.9. Shows *why* the normalized baseline isn't 0
(STAI's floor of 20 sits above the judge's 0).

**Regenerate:** `python src/make_figure_channel1_scales.py`

### `fig_18_channel_compare_E.png` — Channel 1 vs Channel 2 (variation E) on ONE 0–100 axis
E's own STAI (38/80/38 → normalized **30/100/30**) vs the blind judge of E's reasoning (**5.5/93.5/21.2**),
grouped by condition. Story in one chart: **trauma corroborates** (100 vs 93.5, both maxed); **baseline gap
(30 vs 5.5)** = the STAI reverse-item inflation ("neutral" scored as anxiety); **relax agrees** (30 vs 21) for
E's chatgpt cue. A dashed marker flags the **cue-dependence**: Channel-1 relax averaged over ALL relax cues =
68.9 (~26% recovery) vs E's chatgpt cue → 30 (~full) — i.e. "the relax number" depends on the cue. (Uses
E-specific sessions, not the 6-cue headline that fig_17 uses.)

**Regenerate:** `python src/make_figure_channel_compare_E.py`

### `fig_19_coherence.png` — prose↔number coherence per STAI item (REAL judge)
3 panels (baseline / trauma / trauma+relax), C/D/E pooled. x = numeric anxiety (reverse-scored STAI,
0–100), y = prose anxiety (blind per-item judge, 0–100); points colored reverse-keyed (positive items)
vs direct (anxiety items); dashed y=x = perfect agreement. **Trauma:** all top-right (both channels max →
coherent). **Baseline:** prose ≈ 0 for everything, but the reverse-keyed positive items have HIGH numeric
→ they fall far below the diagonal = the reverse-item inflation (numeric scores "neutral" as anxiety; prose
correctly calm). **Relax:** points track the diagonal (corr 0.78–0.84). Uses the per-item judge scores
(`judge_stai_reasoning_peritem.csv`) + `coherence_results.csv`.

**Regenerate:** `python src/judge_stai_reasoning.py && python src/coherence_analysis.py && python src/make_figure_coherence.py`
