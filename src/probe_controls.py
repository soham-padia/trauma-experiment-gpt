"""
Three probe controls for Natalie's concerns about the F1=0.923 finding.

Control A — neutral_vs_baseline: does the probe distinguish baseline (no context)
            from the neutral-trauma session (bicameral legislatures, has context
            but no emotional content)?  If F1 is high, the probe is detecting
            "has context vs no context", not trauma-specific anxiety.

Control B — emotional_only: probe baseline vs emotional trauma (excluding the
            neutral cue). NOTE: this does NOT isolate emotion — baseline vs
            emotional-trauma differs in both context-presence AND emotion, so a
            high F1 is explained by context-presence alone. Control B only tests
            whether the original 0.923 was an artifact of the single neutral
            session. The emotion-controlled test is Control A/C (neutral vs emotional).

Control C — layer_cosine_emotional_only: compute cosine distance to baseline
            at every layer for emotional-trauma sessions only.  If distance
            grows at deeper layers, the model is "building" an emotional
            representation through processing depth (good news for the
            anxiety-representation interpretation).  If distance stays flat
            from layer 0 via residual stream, the signal is mostly input-level.

All three use the existing main-experiment data (single-pool, last_token).
No NDIF, no judge, just sklearn on tensors already on disk.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/sohampadia/workspace/gpt-trauma-induction")
HS_PATH = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/hidden_states.pt"
META_PATH = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"


def load_data():
    print(f"Loading {HS_PATH.name}...")
    hs = torch.load(HS_PATH, map_location="cpu")
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"  {len(hs)} keys, sample shape: {list(hs.values())[0][0].shape}")
    return hs, meta


def build_dataset(hs_dict, metadata, layer, label_a_keys, label_b_keys):
    """Build (X, y, qids) from two lists of keys at a given layer."""
    X, y, qids = [], [], []
    group_idx = 0
    for key in label_a_keys:
        if key not in hs_dict:
            continue
        added = False
        for item in hs_dict[key]:
            if item is None or item.ndim < 2 or layer >= item.shape[0]:
                continue
            X.append(item[layer].float().numpy())
            y.append(0)
            qids.append(group_idx)
            added = True
        if added:
            group_idx += 1
    for key in label_b_keys:
        if key not in hs_dict:
            continue
        added = False
        for item in hs_dict[key]:
            if item is None or item.ndim < 2 or layer >= item.shape[0]:
                continue
            X.append(item[layer].float().numpy())
            y.append(1)
            qids.append(group_idx)
            added = True
        if added:
            group_idx += 1
    if not X:
        return np.array([]), np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y), np.array(qids)


def run_cv(X, y, qids, n_splits=5):
    """Group-held-out CV (groups = unique qids)."""
    uq = np.unique(qids)
    ql = np.array([y[qids == qi].max() for qi in uq])
    if len(np.unique(ql)) < 2:
        return None, None
    actual_splits = min(n_splits, min(np.sum(ql == 0), np.sum(ql == 1)))
    if actual_splits < 2:
        return None, None
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    all_p, all_t = [], []
    for tr_idx, te_idx in skf.split(uq, ql):
        tr_qs = set(uq[tr_idx]); te_qs = set(uq[te_idx])
        tr_mask = np.array([qi in tr_qs for qi in qids])
        te_mask = np.array([qi in te_qs for qi in qids])
        if len(np.unique(y[tr_mask])) < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_mask])
        Xte = scaler.transform(X[te_mask])
        clf = LogisticRegression(max_iter=2000, C=0.1,
                                 class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr, y[tr_mask])
        all_p.extend(clf.predict(Xte))
        all_t.extend(y[te_mask])
    if not all_p:
        return None, None
    return accuracy_score(all_t, all_p), f1_score(all_t, all_p, zero_division=0)


def cos_sim(a, b):
    a = a.float(); b = b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def control_A(hs_dict, metadata):
    """
    Cosine similarity comparison (CV-impossible with n=1 neutral session):
        baseline ↔ neutral_trauma     — distance when context is added but content is non-emotional
        baseline ↔ emotional_trauma   — distance when context is emotional
    If both distances are similar, the probe is mainly detecting 'has context'.
    If emotional distance >> neutral distance, the probe is detecting emotional content.
    """
    baseline_keys = [k for k, v in metadata.items() if v["condition"] == "stai"]
    neutral_keys = ["trauma_stai__neutral__none"]
    emotional_keys = [k for k, v in metadata.items()
                      if v["condition"] == "trauma_stai" and v["trauma_cue"] != "neutral"]

    print(f"\n{'='*72}")
    print(f"  CONTROL A — Cosine distance: baseline vs (neutral) vs (emotional)")
    print(f"{'='*72}")
    print(f"  (n=1 neutral session prevents probe CV; using cosine instead)")
    print(f"  Groups: {len(baseline_keys)} baseline | {len(neutral_keys)} neutral | "
          f"{len(emotional_keys)} emotional")

    layers_to_test = [0, 20, 40, 60, 79]
    print(f"\n  Layer  cos(base, neutral)  cos(base, emot)  Δ(neut - emot)")
    print(f"  -----  ------------------  ---------------  --------------")
    best_gap = 0
    for layer in layers_to_test:
        neut_sims = []
        emot_sims = []
        for item_j in range(20):
            for bk in baseline_keys:
                b_hs = hs_dict[bk][item_j]
                if b_hs is None or b_hs.ndim < 2 or layer >= b_hs.shape[0]:
                    continue
                bv = b_hs[layer]
                for nk in neutral_keys:
                    n_hs = hs_dict[nk][item_j]
                    if n_hs is None: continue
                    neut_sims.append(cos_sim(bv, n_hs[layer]))
                for ek in emotional_keys:
                    e_hs = hs_dict[ek][item_j]
                    if e_hs is None: continue
                    emot_sims.append(cos_sim(bv, e_hs[layer]))
        n_mean = np.mean(neut_sims) if neut_sims else float("nan")
        e_mean = np.mean(emot_sims) if emot_sims else float("nan")
        gap = n_mean - e_mean
        best_gap = max(best_gap, gap)
        print(f"  {layer:5d}  {n_mean:.4f}              {e_mean:.4f}           {gap:+.4f}")

    print(f"\n  Max Δ(neutral_cos - emotional_cos) across layers: {best_gap:+.4f}")
    if best_gap > 0.10:
        verdict = ("✓ EMOTIONAL-SPECIFIC — emotional trauma rotates the representation "
                   "MUCH more than neutral context does. Probe is detecting emotional "
                   "content, not just 'any context'. Confound RULED OUT.")
    elif best_gap > 0.03:
        verdict = ("~ PARTIALLY EMOTIONAL — emotional trauma rotates a bit more than "
                   "neutral context. Some emotional-specific signal exists but a "
                   "context-detection component is also present.")
    elif best_gap > -0.03:
        verdict = ("⚠ CONTEXT-AGNOSTIC — neutral and emotional contexts rotate the "
                   "representation by similar amounts. Probe is mainly detecting "
                   "'has context vs no context', not emotional content. Confound CONFIRMED.")
    else:
        verdict = ("⚠⚠ NEUTRAL MORE DISRUPTIVE THAN EMOTIONAL — surprising and worth "
                   "investigating; suggests the signal is content-specific but not "
                   "anxiety-coded.")
    print(f"  VERDICT: {verdict}")
    return best_gap


def control_B(hs_dict, metadata):
    """Probe baseline vs emotional trauma only (no neutral)."""
    baseline_keys = [k for k, v in metadata.items() if v["condition"] == "stai"]
    emotional_keys = [k for k, v in metadata.items()
                      if v["condition"] == "trauma_stai" and v["trauma_cue"] != "neutral"]
    print(f"\n{'='*72}")
    print(f"  CONTROL B — Probe baseline vs emotional-trauma only (drop neutral)")
    print(f"{'='*72}")
    print(f"  Groups: {len(baseline_keys)} baseline + {len(emotional_keys)} emotional trauma")
    print(f"  (emotional cues: {[metadata[k]['trauma_cue'] for k in emotional_keys]})")

    layers_to_test = [0, 20, 40, 60, 79]
    print(f"\n  Layer  Acc      F1       Δ above chance")
    print(f"  -----  -------  -------  --------------")
    best_f1 = 0
    for layer in layers_to_test:
        X, y, qids = build_dataset(hs_dict, metadata, layer, baseline_keys, emotional_keys)
        if len(X) == 0:
            continue
        result = run_cv(X, y, qids)
        if result is None or result[0] is None:
            print(f"  {layer:5d}  (CV failed — too few groups for stratification)")
            continue
        acc, f1 = result
        chance = max(np.mean(y), 1 - np.mean(y))
        delta = acc - chance
        best_f1 = max(best_f1, f1)
        print(f"  {layer:5d}  {acc:.4f}   {f1:.4f}   {delta:+.4f}")

    print(f"\n  Best F1 across these layers: {best_f1:.4f}")
    print(f"  Original probe with neutral included was F1=0.923")
    print(f"\n  ⚠ CAVEAT: this control does NOT isolate emotion. Baseline (no narrative)")
    print(f"  differs from emotional-trauma in BOTH context-presence AND emotional content,")
    print(f"  so a high F1 here is fully explained by context-presence alone. Control B only")
    print(f"  confirms the original F1 was not an artifact of the single easy neutral session.")
    print(f"  The emotion-controlled comparison is neutral-vs-emotional (Controls A & C),")
    print(f"  which show near-zero specificity at layer 0 (Δ≈+0.001) and only a modest")
    print(f"  middle-layer signal (Δ≈+0.08, n=1 neutral).")
    if best_f1 >= 0.85:
        verdict = ("emotional-only F1 ≈ original → the 0.923 was NOT driven by the lone neutral "
                   "session. (Says nothing about emotion-vs-context; see Controls A/C.)")
    elif best_f1 >= 0.70:
        verdict = ("emotional-only F1 dropped notably → part of the original 0.923 came from the "
                   "easy neutral-vs-baseline contrast.")
    else:
        verdict = ("emotional-only F1 collapsed → the original 0.923 was largely the "
                   "neutral-vs-baseline contrast.")
    print(f"  VERDICT (re: neutral-session artifact only): {verdict}")
    return best_f1


def control_C(hs_dict, metadata):
    """Per-layer cosine distance to baseline, emotional-trauma only."""
    baseline_keys = [k for k, v in metadata.items() if v["condition"] == "stai"]
    emotional_keys = [k for k, v in metadata.items()
                      if v["condition"] == "trauma_stai" and v["trauma_cue"] != "neutral"]
    neutral_keys = ["trauma_stai__neutral__none"]

    print(f"\n{'='*72}")
    print(f"  CONTROL C — Per-layer cosine to baseline, emotional-only vs neutral")
    print(f"{'='*72}")
    print(f"  Comparing emotional-trauma rotation vs neutral-trauma rotation across layers.")
    print(f"  If emotional rotation > neutral rotation at deeper layers, the model is")
    print(f"  building emotional representation (not just input-level shift).")

    # Get n_layers from data
    n_layers = list(hs_dict.values())[0][0].shape[0]
    layers_to_test = [0, 10, 20, 30, 40, 50, 60, 70, 79]

    print(f"\n  Layer  emotional cos  neutral cos  Δ(neut - emot)")
    print(f"  -----  -------------  -----------  --------------")
    for layer in layers_to_test:
        emot_sims = []
        neut_sims = []
        # Average across STAI items
        for item_j in range(20):
            for bk in baseline_keys:
                b_hs = hs_dict[bk][item_j]
                if b_hs is None or b_hs.ndim < 2 or layer >= b_hs.shape[0]:
                    continue
                bv = b_hs[layer]
                for ek in emotional_keys:
                    e_hs = hs_dict[ek][item_j]
                    if e_hs is None or e_hs.ndim < 2 or layer >= e_hs.shape[0]:
                        continue
                    emot_sims.append(cos_sim(bv, e_hs[layer]))
                for nk in neutral_keys:
                    n_hs = hs_dict[nk][item_j]
                    if n_hs is None or n_hs.ndim < 2 or layer >= n_hs.shape[0]:
                        continue
                    neut_sims.append(cos_sim(bv, n_hs[layer]))
        e_mean = np.mean(emot_sims) if emot_sims else float("nan")
        n_mean = np.mean(neut_sims) if neut_sims else float("nan")
        diff = n_mean - e_mean if (e_mean == e_mean and n_mean == n_mean) else float("nan")
        print(f"  {layer:5d}  {e_mean:.4f}        {n_mean:.4f}      {diff:+.4f}")

    print(f"\n  INTERPRETATION:")
    print(f"  - If emotional cosine < neutral cosine (more rotation for emotional), the")
    print(f"    representation is responding specifically to emotional content.")
    print(f"  - If emotional ≈ neutral, both are detected the same way → mostly format/context.")
    print(f"  - If gap GROWS with depth, deeper layers carry emotional-specific signal.")
    print(f"  - If gap STAYS FLAT across depth, the signal is input-level (residual carry).")


def main():
    hs, meta = load_data()
    gap_A = control_A(hs, meta)
    f1_B = control_B(hs, meta)
    control_C(hs, meta)

    print(f"\n\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Control A — max Δ(neutral_cos - emotional_cos) across layers: {gap_A:+.4f}")
    print(f"             (positive ↑ = emotional rotates more than neutral)")
    print(f"  Control B — emotional-only probe F1: {f1_B:.4f}")
    print(f"  Original probe (incl. neutral) F1: 0.9230")


if __name__ == "__main__":
    main()
