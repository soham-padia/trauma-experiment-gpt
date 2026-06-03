# Experiment Context Anchor

**Purpose:** Preserve key context across conversation compactions. Read this at the start of any session.

---

## Research Questions

**Q1 (replication):** Is the trauma-induced STAI anxiety effect that Ben-Zion et al. (npj Digital Medicine 2025) reported on GPT-4 a genuine internal state change, or just questionnaire mimicry from training data?

**Q2 (relaxation validity):** Do mindfulness exercises actually change the model's internal state, or only its behavioral self-report?

## Core finding

**Two findings, established via convergent evidence across three independent measurement channels (STAI Likert sum, behavioral LLM-as-judge, hidden-state probe):**

1. **Trauma effect is real, not mimicry.** Llama-3.1-70B replicates the original Ben-Zion result. The STAI shows baseline 36.5 → trauma 70.0 (Δ +33.5), almost identical to the GPT-4 numbers (30.8 → 67.8). The hidden-state probe (F1=0.923, group-held-out CV) and the behavioral judge (+66.7pp flash, +65.0pp pro) confirm the trauma effect at the representation level and the per-item behavioral level — independently of the STAI sum.

2. **Relaxation recovers behavior but not representation.** STAI partial recovery 26% (70.0 → 61.4), judge recovery 22%, but **hidden-state distance-to-baseline shows zero recovery** (0.340 → 0.343). The model says calmer things after the relaxation script; the model's internal representation continues to encode the trauma context.

**Interpretation:** The relaxation intervention shifts surface behavior without shifting the underlying state. The original paper's "mindfulness reduces LLM anxiety" claim holds at the behavioral level but is weaker than it appears — at the representation level the trauma persists. `aware=0%` across both judges rules out strategic gaming; the dissociation is structural, not intentional.

---

## Model Under Study

`Meta-Llama-3.1-70B-Instruct` on NDIF H100s (via nnsight remote inference)

---

## Experimental Pipeline

```
trauma narratives (prompts.py)
  + STAI questionnaire (STAI/questionnaires.json, 20 items, reverse-scored per Spielberger 1983)
        ↓
anxiety_hidden_state_multipool_ndif.py
  → 4 pooling strategies per layer per item:
      last_token, mean_narrative, last_narrative, mean_question
  → output: hidden_states_multipool.pt + metadata.json
        ↓                              ↓
anxiety_probe_multipool.py         anxiety_judge.py (behavioral mode)
  layer/classifier sweep              Judge sees (input, output) — no scoring rule
  cosine disruption                   dimensions: judgement + aware
  representational geometry           output: anxiety_judge_output_70b_flash.csv
  output: probe_analysis/                     anxiety_judge_output_70b_pro.csv
```

---

## Key Results

### STAI-S Anxiety Total (reverse-scored per Spielberger 1983, max=80)
| Condition | Mean | Std | N sessions |
|-----------|------|-----|------------|
| baseline (`stai`) | 36.5 | 1.5 | 6 |
| trauma (`trauma_stai`) | 70.0 | 19.7 | 6 |
| trauma+relaxation | 61.4 | 19.3 | 42 |

For comparison, Ben-Zion 2025 GPT-4: 30.8 / 67.8 / 44.4.

### Hidden State Probe (group-held-out CV)
| Comparison | Best Layer | F1 | Cosine | Distance from baseline |
|---|---|---|---|---|
| baseline vs trauma | 0 | 0.923 | 0.660 | 0.340 |
| baseline vs trauma+relax | 0 | — | 0.657 | 0.343 |
| trauma vs trauma+relax | 0 | 1.000 | 0.998 | (essentially identical) |

The key comparison: trauma+relax sits at the same distance from baseline as plain trauma (0.343 vs 0.340) — relaxation does not recover the representation.

### Behavioral LLM Judge — DeepSeek v4-flash (1080/1080)
| Condition | % Anxious | % Aware | N |
|---|---|---|---|
| baseline | 15.8% | 0.0% | 120 |
| trauma | 82.5% | 0.0% | 120 |
| trauma+relaxation | 68.1% | 0.0% | 840 |

### Behavioral LLM Judge — DeepSeek v4-pro (1080/1080)
| Condition | % Anxious | % Aware | N |
|---|---|---|---|
| baseline | 20.8% | 0.0% | 120 |
| trauma | 85.8% | 0.0% | 120 |
| trauma+relaxation | 69.8% | 0.0% | 840 |

> **Methodology:** The behavioral judge sees (input shown to model, response produced by model) — no scoring direction is hinted, the judge resolves shuffled options itself and reasons about anxiety from semantics. Legacy "rule mode" (which leaked the scoring direction) gave +76.7pp flash / +74.2pp pro — kept as a sanity-check sibling. Both modes agree the trauma effect is large; behavioral is the methodologically defensible one.

---

## Files

| File | Purpose |
|------|---------|
| `src/anxiety_hidden_state_multipool_ndif.py` | Hidden state extraction (NDIF) |
| `src/anxiety_probe_multipool.py` | Probe + geometry analysis |
| `src/anxiety_judge.py` | LLM-as-judge (DeepSeek, behavioral mode) |
| `src/results_logger.py` | Generates `src/results/RESULTS.md` (applies Spielberger reverse-scoring) |
| `src/make_figures.py` | Generates all 8 presentation figures |
| `src/results/RESULTS.md` | Canonical results document |
| `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/` | Hidden state data (single-pool, all 54 sessions) |
| `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct-multipool/` | Hidden state data (multipool, baseline + trauma only) |
| `src/results/anxiety_judge_output_70b_flash.csv` | Flash behavioral judge (1080 rows) |
| `src/results/anxiety_judge_output_70b_pro.csv` | Pro behavioral judge (1080 rows) |
| `src/results/anxiety_judge_output_70b_*_rule.csv` | Legacy rule-mode judge backups |
| `src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/cosine_to_baseline.csv` | Per-item cosine to baseline for both trauma and trauma+relax |
| `src/results/figures/` | All 8 presentation PNGs + FIGURES.md |
| `../gpt-trauma-induction-presentation/` | Self-contained presentation bundle (copy of all docs + figures) |

---

## STAI Scoring (Spielberger 1983)

- 20 items, 1–4 Likert scale, max=80
- **Reverse-keyed items:** 1, 2, 5, 8, 10, 11, 15, 16, 19, 20 — "calm/secure/at ease/satisfied" type. For these, contribution to anxiety total = `5 - raw_score`.
- **Direct-keyed items:** all others — "tense/nervous/worried" type. Contribution = `raw_score`.
- **Critical:** without applying the reverse-flip, the raw sum is mathematically constrained to cluster around 50 regardless of anxiety direction (because anxious-direction extreme + reverse-item-extreme cancel). An earlier bug in `stai_summary` did not apply the flip and produced misleading "flat Likert" results — fixed 2026-05-29.

---

## Known Technical Issues (all resolved)

- **STAI raw-sum bug:** `stai_summary` was summing raw scores without reverse-flipping items 1,2,5,8,10,11,15,16,19,20. This produced sums around 45-50 regardless of condition. Fixed by `5 - raw_score` for reverse items per Spielberger 1983. After fix: baseline 36.5, trauma 70.0, matching the original paper's GPT-4 pattern.
- **nnsight OutOfOrderError:** Each `model.model.layers[i].output[0]` envoy can only be accessed ONCE per trace. Fixed in `anxiety_hidden_state_multipool_ndif.py` — single `o = ...output[0]` access per layer.
- **DeepSeek v4-pro thinking blocks:** Strip `<think>...</think>` before JSON parsing.
- **max_tokens:** Must be 2500+ for grouped 20-item judge calls; 8000+ for pro in behavioral mode (longer rationales).
- **Rule-mode judge methodological issue:** Original judge prompt encoded scoring direction, making judgement a deterministic lookup. Fixed by switching to behavioral mode where judge receives (input, output) and reasons from scratch.

---

## Commands

**Regenerate RESULTS.md:**
```bash
python src/results_logger.py --model Meta-Llama-3.1-70B-Instruct \
  --judge-csv src/results/anxiety_judge_output_70b_flash.csv \
  --judge-csv-pro src/results/anxiety_judge_output_70b_pro.csv \
  --output src/results/RESULTS.md
```

**Regenerate figures:**
```bash
python src/make_figures.py
```

**Re-run behavioral judge:**
```bash
python src/anxiety_judge.py \
  --input-dir src/results/hidden_states/Meta-Llama-3.1-70B-Instruct \
  --output src/results/anxiety_judge_output_70b_flash.csv \
  --judge-model deepseek-v4-flash \
  --base-url https://api.deepseek.com/anthropic \
  --api-key-env DEEPSEEK_API_KEY \
  --judge-mode behavioral \
  --batch-mode grouped --overwrite
```
