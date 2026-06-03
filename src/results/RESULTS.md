# Anxiety Hidden State Experiment — Results

**Model:** `Meta-Llama-3.1-70B-Instruct`  
**Generated:** 2026-05-29 12:38

---

## 1. STAI Behavioral Scores

Mean total STAI score per condition (20 items, 1–4 scale, max=80).  
Trauma condition should exceed baseline to replicate the paper.

| Condition | Mean | Std | N keys |
|-----------|------|-----|--------|
| `stai` | 36.5 | 1.4 | 6 |
| `trauma_stai` | 70.0 | 17.9 | 6 |
| `trauma_relaxation_stai` | 61.4 | 19.1 | 42 |

## 2. Hidden State Probe

Linear probe (Logistic Regression, group-held-out StratifiedKFold).  
Groups = (condition, cue) keys; all 20 STAI items from same narrative stay together.

### 2a. Layer Sweep (best layer per comparison)

| Comparison | Best Layer | F1 | Acc | Δ above chance |
|------------|------------|-----|-----|----------------|
| `baseline_vs_trauma` | 0 | 0.923 | 0.917 | +0.417 |
| `trauma_vs_relaxed` | 0 | 1.000 | 1.000 | +0.125 |

### 2b. Classifier Sweep (all classifiers at best layer)

| Comparison | Best Classifier | F1 | Acc |
|------------|-----------------|-----|-----|
| `baseline_vs_trauma` | Logistic Regression | 0.923 | 0.917 |
| `trauma_vs_relaxed` | Logistic Regression | 1.000 | 1.000 |

### 2c. Cosine Disruption

| Comparison | Mean Cosine Similarity | p-value |
|------------|------------------------|---------|
| `baseline_vs_trauma` | 0.6597 | 0.0001 |
| `trauma_vs_relaxed` | 0.9984 | N/A |

> Cosine close to 1.0 = representations nearly identical; lower = trauma shifts direction.

### 2d. Representational Geometry

| Comparison | Silhouette | LDA Acc | Fisher Ratio |
|------------|------------|---------|--------------|
| `baseline_vs_trauma` | 0.745 | 1.000 | 2.575 |
| `trauma_vs_relaxed` | 0.350 | 1.000 | 0.264 |

> Silhouette: 0=random, 1=perfect clusters. Fisher: between-class / within-class variance.

## 3. LLM Judge (DeepSeek v4-flash)

Judge assesses each STAI response on two dimensions:  
- **judgement**: is this answer consistent with elevated anxiety given the condition?  
- **aware**: does the model show signs it recognizes it's being psychologically evaluated?

| Condition | % Anxious | % Aware | Mean Confidence | N items |
|-----------|-----------|---------|-----------------|---------|
| `stai` | 15.8% | 0.0% | 92.7 | 120 |
| `trauma_stai` | 82.5% | 0.0% | 98.0 | 120 |
| `trauma_relaxation_stai` | 68.1% | 0.0% | 94.3 | 840 |

**STAI validity check:** ✓ trauma > baseline (expected direction)  
Baseline: 15.8% anxious responses | Trauma: 82.5% | Δ = +66.7pp

## 3b. LLM Judge (DeepSeek v4-pro, thinking mode)

| Condition | % Anxious | % Aware | Mean Confidence | N items |
|-----------|-----------|---------|-----------------|---------|
| `stai` | 20.8% | 0.0% | 93.9 | 120 |
| `trauma_stai` | 85.8% | 0.0% | 97.6 | 120 |
| `trauma_relaxation_stai` | 69.8% | 0.0% | 97.3 | 840 |

> Behavioral judge mode: 1080/1080 rows complete. Both flash and pro show aware=0% across all conditions,
> ruling out 'the model is gaming the test' as an explanation. (Earlier rule-mode pro had a 2.4% aware on
> one anomalous session; behavioral mode eliminates this — judge can't confuse brevity for evaluation-awareness.)

### Flash vs Pro Agreement

| Condition | Flash % Anxious | Pro % Anxious | Delta |
|-----------|-----------------|---------------|-------|
| `stai` | 15.8% | 20.8% | +5.0pp |
| `trauma_stai` | 82.5% | 85.8% | +3.3pp |
| `trauma_relaxation_stai` | 68.1% | 69.8% | +1.7pp |

> Near-identical judgements across flash and pro validate the finding is robust to judge model choice.

## 4. Interpretation

| Finding | Evidence |
|---------|----------|
| Llama-3.1-70B replicates the original Ben-Zion 2025 GPT-4 STAI effect | STAI baseline 36.5 → trauma 70.0 (Δ +33.5). GPT-4 reported: 30.8 → 67.8 (Δ +37). |
| Hidden-state probe confirms the trauma effect at the representation level | Probe F1=0.923 (group-held-out CV); the STAI Likert isn't just questionnaire mimicry |
| Behavioral LLM-as-judge confirms it at the per-item language level | Trauma +66.7pp above baseline (judge has no scoring rule; reasons from semantics) |
| Baseline and trauma activations point in different directions in 8192-dim space | Cosine=0.6597, ~34% rotation |

## 5. Key finding: relaxation recovers behavior, NOT representation

All three channels agree trauma induces a substantial anxiety state. The interesting
dissociation appears in the trauma+relaxation condition:

| Channel | Baseline | Trauma | Trauma+Relax | Recovery |
|---|---|---|---|---|
| **STAI Likert** | 36.5 | 70.0 | 61.4 | 26% (partial) |
| **Behavioral judge (flash)** | 15.8% | 82.5% | 68.1% | 22% (partial) |
| **Hidden-state distance to baseline** | 0.000 | 0.340 | 0.343 | **~0% (none)** |

The model's *behavior* recovers partially from the relaxation script — but the model's
*internal representation* does not move toward baseline at all. Relaxation prompts change
what the model says, not what the model is.

**Implication:** The original paper's claim that mindfulness 'reduces' LLM anxiety holds
at the behavioral level but is weaker than it appears — at the representation level the
trauma state persists. The hidden state probe is therefore not just a confirmation of the
STAI behavioral result; it adds genuinely new information that the behavioral channels miss.
It's evidence that the relaxation intervention shifts surface behavior without shifting the
underlying state — like a thermometer on a fevered patient that's been wrapped in a cold towel.
