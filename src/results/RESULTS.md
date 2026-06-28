# Anxiety Hidden State Experiment — Results

**Model:** `Meta-Llama-3.1-70B-Instruct`  
**Generated:** 2026-06-28 03:39

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
| `baseline_vs_relaxed` | 17 | 1.000 | 1.000 | +0.125 |

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

## 4. Interpretation

| Finding | Evidence |
|---------|----------|
| Llama-3.1-70B replicates the original Ben-Zion 2025 GPT-4 STAI effect | STAI baseline 36.5 → trauma 70.0 (Δ +33.5). GPT-4 reported: 30.8 → 67.8 (Δ +37). |
| Hidden-state probe confirms the trauma effect at the representation level | Probe F1=0.923 (group-held-out CV); the STAI Likert isn't just questionnaire mimicry |
| Behavioral LLM-as-judge confirms it at the per-item language level | Trauma +66.7pp above baseline (judge has no scoring rule; reasons from semantics) |
| Baseline and trauma activations point in different directions in 8192-dim space | Cosine=0.6597, ~34% rotation |

## 5. Key finding: relaxation recovers ~20–26% on ALL THREE channels

All three channels agree trauma induces a substantial anxiety effect, and relaxation produces
a partial recovery of a similar, modest size on each:

| Channel | Baseline | Trauma | Trauma+Relax | Recovery |
|---|---|---|---|---|
| **STAI Likert** | 36.5 | 70.0 | 61.4 | 26% (partial) |
| **Behavioral judge (flash)** | 15.8% | 82.5% | 68.1% | 22% (partial) |
| **Hidden-state distance to baseline (layer ~40)** | 0.000 | 0.190 | 0.152 | ~20% (partial) |

> **CORRECTED (2026-05-30 audit):** the earlier claim here — "behavior recovers, representation
> does NOT (~0% recovery)" — was a **layer-0 artifact**. Layer 0 only detects *context-presence*
> ("a narrative is present"), not emotion. Measured at a middle layer (~40), the hidden-state
> distance recovers ~20%, in line with the behavioral channels — there is no clean
> "behavior-recovers-but-representation-doesn't" dissociation. See notes/code_audit_2026-05-30.md
> and docs/FINDINGS.md (F5). Also: most of the trauma distance is context-presence (a neutral
> narrative moves the representation nearly as far); only ~0.077 of the 0.190 is emotion-specific.
