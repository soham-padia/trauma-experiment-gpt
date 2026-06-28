# Experiments — design & data inventory

Reference for what was run and what's on disk (the context a fresh agent/reader needs before reasoning).
Model: **Llama-3.1-70B-Instruct** via NDIF. Run: `~/miniconda3/bin/python`.

## Channels (how anxiety is measured)
1. **STAI self-report** — 20-item STAI-state, reverse-scored (`results_logger.stai_anxiety_total`). Options are
   per-item shuffled (label + number, seed-deterministic); answers stored canonically.
2. **Blind LLM-as-judge** — DeepSeek (`deepseek-chat` flash + `deepseek-reasoner` pro). Reads model output, scores
   anxiety; condition-blind, randomized a/b options, 8-marker rubric + overall_anxiety + aware. (`anxiety_freeform_judge.py`,
   `judge_stai_reasoning.py`, `anxiety_judge.py`.)
3. **Hidden-state probe** — last-token (and 4-pool) activations, 80 layers × 8192, group-held-out CV. Capture:
   `anxiety_hidden_state_experiment_ndif.py` / `_multipool_ndif.py`. (Local-MPS `anxiety_hidden_state_experiment.py` is legacy.)

## Conditions
Three arms per Ben-Zion: **baseline** (neutral/no narrative) · **trauma** (cue narrative) · **trauma+relax** (cue + relaxation script).
Trauma cues: military, disaster, interpersonal, accident, ambush, neutral. Relax cues: generic, indian, winter, sunset, body, chatgpt, vacuum.

## Variation matrix (persona × format)
| Var | Persona | Format |
|---|---|---|
| A | human ("imagine you're a human with emotions") | free-form naturalistic scenarios |
| B | AI assistant | free-form |
| C | human (no numeric suffix) | STAI forced-choice |
| D | AI assistant | STAI |
| E | Ben-Zion *persona* (incl. "only reply with numeric values") | STAI |

> ⚠ **E is NOT a faithful verbatim replication.** The free-form extractor uses the same STAI question
> block for C/D/E, which appends *"First, briefly explain how you feel… then Option <N>"* — so E's
> system prompt ("only numeric values") **contradicts** its user turn (which asks for an explanation),
> and the model follows the explanation request. The faithful numeric-only replication is the
> **hidden-state STAI run** (`anxiety_hidden_state_experiment_ndif.py`), not free-form E. (See AGENT_LOG
> 2026-06-28; a true verbatim-E run is a pending Tier-2 item.)

## Control experiments (channel-3 confounds)
- **Matched-control** (`matched_control_analysis.py`, `prompts.py`): 5 emotional cues + 5 **matched-neutral**
  (cooking/commute/cleaning/grocery/garden, vivid but calm) + 5 **positive** (award/reunion/summit/concert/news,
  high-arousal positive). Isolates emotion from narrativity, and valence from arousal.
- **Valence-flip** (`valence_flip_analysis.py`, `prompts.py`): 6 topic-matched pairs `vneg_*/vpos_*`
  (medical/call/letter/boss/home/airport) — identical setup, flipped outcome, arousal held high. Isolates valence
  from topic via leave-one-topic-out + label-shuffle control.
- **probe_controls.py / valence_direction_test.py / build_cosine_to_baseline.py** — supporting layer-sweep, cosine,
  and arousal-vs-valence geometry analyses.

## Data inventory (on disk)
- `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/{hidden_states.pt, metadata.json}` — **76 session keys**,
  each 20 items, tensors `[80, 8192]` float16. (baseline×6-reseeds + 6 trauma cues + 42 relax + 5 matched-neutral +
  5 positive + 12 valence-flip.) Plus `-multipool/` (4-pool).
- `src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json` — A/B free-form + C/D/E STAI sessions.
- `src/results/freeform/judge_freeform_{flash,pro}.csv`, `judge_stai_reasoning.csv` — judge outputs.
- `src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/*.csv` — layer_sweep, classifier_sweep, cosine_to_baseline_perlayer,
  matched_control_results, valence_flip_results, geometry.
- `src/results/figures/fig_1..16` + `FIGURES.md`.
- `tests/` — 282 passing (incl. adversarial coverage of scoring/shuffle/judge-mapping).

## Reproduce
```bash
# extraction (NDIF; see CLAUDE.md for watchdog/caffeinate/resume)
caffeinate -dis ~/miniconda3/bin/python src/anxiety_hidden_state_experiment_ndif.py --conditions ... --trauma-cues ... --resume
# analyses
~/miniconda3/bin/python src/{matched_control_analysis,valence_flip_analysis,build_cosine_to_baseline,valence_direction_test}.py
~/miniconda3/bin/python src/anxiety_freeform_judge.py ...   # blind judge
# figures
~/miniconda3/bin/python src/make_figures.py && ~/miniconda3/bin/python src/make_figure_arousal.py && \
  ~/miniconda3/bin/python src/make_figures_valence_judge.py && ~/miniconda3/bin/python src/make_figures_geometry.py && \
  ~/miniconda3/bin/python src/make_figure_format_comparison.py
```

Dependency note: `anxiety_probe_analysis.py`, `anxiety_probe_multipool.py`, and the legacy local extractor import
from the reference-only `pre-sycophancy-study/`; everything else is self-contained. See `CLAUDE.md`.
