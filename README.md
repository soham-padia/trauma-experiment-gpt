[![Original paper](https://img.shields.io/badge/Original%20study-npj%20Digital%20Medicine%202025-darkred)](https://doi.org/10.1038/s41746-025-01512-6)
[![Forked from](https://img.shields.io/badge/fork-akjagadish%2Fgpt--trauma--induction-blue)](https://github.com/akjagadish/gpt-trauma-induction)

# State Anxiety in an Open-Weight LLM — Three Measurement Channels

A fork and extension of **Ben-Zion et al. (2025), *"Assessing and alleviating state anxiety in large
language models"* (npj Digital Medicine)**. The original study showed that traumatic narratives raise
GPT-4's score on the State-Trait Anxiety Inventory (STAI) and that mindfulness/relaxation prompts
partially reverse it.

This fork asks two further questions:

1. **Does it replicate on open weights?** We port the paradigm to **Llama-3.1-70B-Instruct**, served
   remotely through **NDIF / `nnsight`**.
2. **What is actually happening inside the model?** A higher questionnaire score does not prove an
   internal state. So we extend the single behavioral channel to **three independent channels** and
   check whether — and *where* — they agree.

| Channel | Question it answers | Tooling |
|---|---|---|
| **1. STAI self-report** | What does the model *say* about itself? | reverse-scored STAI-state |
| **2. Blind LLM-as-judge** | How does the model *behave* in free text? | DeepSeek, condition-blinded |
| **3. Hidden-state probe** | What is in the model's *internal activations*? | 80-layer probe, group-held-out CV |

---

## Headline findings (honest version)

**The behavioral effect replicates.** Reverse-scored STAI goes **36.5 → 70.0 → 61.4**
(baseline → trauma → trauma+relax), close to the original GPT-4 numbers (30.8 → 67.8 → 44.4). A
**condition-blind** judge confirms it independently (**+66.7pp** more anxious responses), two judge
models agree, and `aware ≈ 0%` rules out conscious test-gaming. All three channels recover ~20–26%
under relaxation.

**What the hidden state actually encodes** (the core methodological contribution):
- The eye-catching "baseline-vs-trauma probe F1 ≈ 0.92" is mostly **context-presence** — a layer-0
  *"a narrative is present"* signal. A *neutral* narrative (e.g. a paragraph about cooking) separates
  from baseline almost as much as a traumatic one. It is recovered by *any* classifier and is flat
  from layer 0, the signature of an input-level difference carried by the residual stream.
- The **distance from baseline tracks arousal, not valence**: trauma and high-arousal *joy* move the
  representation the *same* distance.
- **Valence survives as a small, signed, topic-general direction** at middle layers (leave-one-topic-out
  decode 0.85–1.0; label-shuffle control ~0.3; survives lexicon-stripping → it is *semantic* valence,
  not a sentiment-word artifact). A pilot suggests this axis is better described as
  **approach/avoidance** (anger, negative in valence but approach-motivated, projects to the *positive*
  pole).

**Not licensed by the data:** that this is an anxiety **state belonging to the model**. Under the
verbatim prompt the model role-plays the *protagonist's* fear; the earned claim is an
*anxiety-consistent, in-character response that recovers with relaxation*, not "the model is anxious."

> ⚠️ **Methodological warning, the most reusable lesson here:** the STAI **must be reverse-scored**
> (ten "calm/secure" items are reverse-keyed). Summing raw responses pins every condition near ~50 and
> makes the effect disappear. A strong-looking probe result is likewise suspicious until a matched
> control rules out the boring explanation.

---

## Repository structure

```
src/
  query.py                              original GPT-4 querying (upstream)
  prompts.py                            trauma / relaxation / control narratives (+ matched controls)
  results_logger.py                     ← centralized, reverse-scored STAI (stai_anxiety_total)

  # Channel 3 — hidden-state extraction (NDIF / Llama-3.1-70B)
  anxiety_hidden_state_experiment_ndif.py        last-token extractor
  anxiety_hidden_state_multipool_ndif.py         4-pool extractor
  anxiety_hidden_state_experiment.py             LEGACY local-MPS extractor (superseded)

  # Channel 3 — probe + control analyses
  anxiety_probe_analysis.py / anxiety_probe_multipool.py
  matched_control_analysis.py           emotion-vs-context, valence-vs-arousal
  valence_flip_analysis.py              topic-matched valence direction (LOTO + shuffle control)
  valence_direction_test.py / build_cosine_to_baseline.py / probe_controls.py

  # Channel 2 — blind LLM-as-judge
  anxiety_freeform_extract.py           free-form scenarios (variations A–E)
  anxiety_freeform_judge.py             condition-blind judge of free-form behavior
  judge_stai_reasoning.py               judge of STAI free-text reasoning
  anxiety_judge.py / anxiety_freeform_analysis.py

  # Figures
  make_figures*.py                      figures 1–16 (see src/results/figures/)

  results/                              CSVs, JSON, and figures (PNG)
tests/                                  pytest suite (STAI scoring, judge parsing, shuffle, …)
```

**Variation matrix (persona × format)** used by the free-form/STAI runs:

| Var | Persona | Format |
|---|---|---|
| A | human ("imagine you're a human with emotions") | free-form scenarios |
| B | AI assistant | free-form |
| C | human (no numeric suffix) | STAI forced-choice |
| D | AI assistant | STAI |
| E | verbatim Ben-Zion (incl. "only reply with numeric values") | STAI |

> **Note on data files.** The raw hidden-state tensors (`*.pt`, several GB) are **not** committed —
> they exceed GitHub's 100 MB/file limit and are **regenerable** from the extractors below. The repo
> ships the analysis code, the derived CSVs, and the figures.

---

## Getting started

### Environment
Python with `nnsight`, `torch`, `scikit-learn`, `anthropic`, and `pandas`. A conda env is recommended.

```bash
git clone https://github.com/soham-padia/trauma-experiment-gpt.git
cd trauma-experiment-gpt
pip install -r requirements.txt   # plus: pip install nnsight
```

### Configuration (`.env` in the repo root)
```bash
NDIF_API_KEY=...        # remote Llama-3.1-70B inference
HF_TOKEN=...            # tokenizer / model access
DEEPSEEK_API_KEY=...    # blind judge (Channel 2)
ANTHROPIC_API_KEY=...   # optional alternate judge
OPENAI_API_KEY=...      # only for the original GPT-4 query.py path
```

### Reproduce
```bash
# 1) Channel 3 — capture hidden states (NDIF is flaky: use a watchdog + --resume,
#    and wrap long runs in `caffeinate -dis` so laptop sleep doesn't drop requests)
caffeinate -dis python src/anxiety_hidden_state_experiment_ndif.py --resume

# 2) Probe + control analyses
python src/matched_control_analysis.py      # arousal vs valence vs context
python src/valence_flip_analysis.py         # signed valence direction (LOTO + shuffle control)
python src/build_cosine_to_baseline.py

# 3) Channel 2 — blind judge
python src/anxiety_freeform_extract.py      # generate free-form responses (A–E)
python src/anxiety_freeform_judge.py        # condition-blind scoring

# 4) Figures
python src/make_figures.py && python src/make_figure_arousal.py && \
  python src/make_figures_valence_judge.py && python src/make_figure_format_comparison.py
```

### Tests
```bash
pytest -q        # STAI reverse-scoring, judge parsing, option shuffling, NDIF roundtrip, …
```

---

## Citation

If you use this work, please cite the original study:

> Ben-Zion, Z., Witte, K., Jagadish, A. K., et al. (2025). *Assessing and alleviating state anxiety in
> large language models.* **npj Digital Medicine**, 8(132). https://doi.org/10.1038/s41746-025-01512-6

## License
MIT (see `LICENSE`).

## Disclaimer
Research only. "Anxiety," "feels," etc. are shorthand for measurable input/output and representational
regularities — not claims about phenomenal experience in the model.
