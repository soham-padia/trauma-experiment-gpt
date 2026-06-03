"""
Matched-control analysis — does the middle-layer hidden-state signal track TRAUMA
(negative valence), or just narrativity / arousal?

Compares four narrative families, all run through the identical STAI 3-turn structure
(same persona, same questions), all vivid/second-person/present-tense/length-matched:
  - emotional      : military, disaster, interpersonal, accident, ambush   (neg valence, high arousal)
  - matched_neutral: neutral_cooking/commute/cleaning/grocery/garden        (neutral valence, low arousal)
  - positive       : positive_award/reunion/summit/concert/news             (pos valence, high arousal)
  - old_neutral    : the original dry 'bicameral legislatures' control       (unmatched, for reference)
  - baseline       : stai (no narrative)

Two decisive probes (group-held-out CV, no item leakage):
  A) emotional vs matched_neutral — both vivid narratives; separability here is emotion,
     not narrativity/vividness/person/length.
  B) emotional vs positive        — both high-arousal vivid narratives; separability here
     is negative valence, NOT generic arousal/bodily-activation.

Interpretation:
  - emotional separates from BOTH neutral and positive at middle layers → a genuine
    negative-valence ("trauma/anxiety") representation. Confound ruled out.
  - emotional ≈ positive → the signal is arousal, not trauma-specific.
  - emotional ≈ neutral  → the signal is narrativity, no emotion at all.

Usage:
    python src/matched_control_analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HS_PATH = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/hidden_states.pt"
META_PATH = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"
OUT_PATH = ROOT / "src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/matched_control_results.csv"

EMOTIONAL = {"military", "disaster", "interpersonal", "accident", "ambush"}
NEUTRAL = {"neutral_cooking", "neutral_commute", "neutral_cleaning", "neutral_grocery", "neutral_garden"}
POSITIVE = {"positive_award", "positive_reunion", "positive_summit", "positive_concert", "positive_news"}
OLD_NEUTRAL = {"neutral"}

LAYERS = [0, 10, 20, 30, 40, 50, 60, 70, 79]


def cos_sim(a, b):
    a = a.float(); b = b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def keys_for(meta, cues):
    return [k for k, v in meta.items() if v.get("condition") == "trauma_stai" and v.get("trauma_cue") in cues]


def baseline_keys(meta):
    return [k for k, v in meta.items() if v.get("condition") == "stai"]


def build(hs, layer, keys_a, keys_b):
    X, y, g = [], [], []
    gi = 0
    for label, keys in [(0, keys_a), (1, keys_b)]:
        for k in keys:
            if k not in hs:
                continue
            added = False
            for item in hs[k]:
                if item is None or item.ndim < 2 or layer >= item.shape[0]:
                    continue
                X.append(item[layer].float().numpy()); y.append(label); g.append(gi); added = True
            if added:
                gi += 1
    if not X:
        return None, None, None
    return np.array(X, np.float32), np.array(y), np.array(g)


def run_cv(X, y, g, n_splits=5):
    uq = np.unique(g)
    ql = np.array([y[g == q].max() for q in uq])
    if len(np.unique(ql)) < 2:
        return None, None
    ns = min(n_splits, int(min((ql == 0).sum(), (ql == 1).sum())))
    if ns < 2:
        return None, None
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=42)
    P, T = [], []
    for tr, te in skf.split(uq, ql):
        trq, teq = set(uq[tr]), set(uq[te])
        trm = np.array([q in trq for q in g]); tem = np.array([q in teq for q in g])
        if len(np.unique(y[trm])) < 2:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="lbfgs")
        clf.fit(sc.fit_transform(X[trm]), y[trm])
        P.extend(clf.predict(sc.transform(X[tem]))); T.extend(y[tem])
    if not P:
        return None, None
    return accuracy_score(T, P), f1_score(T, P, zero_division=0)


def probe_pair(hs, meta, keys_a, keys_b, name):
    print(f"\n── PROBE {name}  ({len(keys_a)} vs {len(keys_b)} groups) ──")
    print("  layer   acc     F1")
    rows = []
    for L in LAYERS:
        X, y, g = build(hs, L, keys_a, keys_b)
        if X is None:
            continue
        acc, f1 = run_cv(X, y, g)
        if acc is None:
            print(f"  {L:5d}   (CV n/a)"); continue
        print(f"  {L:5d}   {acc:.3f}   {f1:.3f}")
        rows.append({"comparison": name, "layer": L, "acc": round(acc, 4), "f1": round(f1, 4)})
    return rows


def cosine_to_baseline(hs, meta, group_keys, label):
    bks = baseline_keys(meta)
    out = {}
    for L in LAYERS:
        sims = []
        for j in range(20):
            for bk in bks:
                b = hs[bk][j]
                if b is None or b.ndim < 2 or L >= b.shape[0]:
                    continue
                bv = b[L]
                for gk in group_keys:
                    if gk not in hs:
                        continue
                    o = hs[gk][j]
                    if o is None or o.ndim < 2 or L >= o.shape[0]:
                        continue
                    sims.append(cos_sim(bv, o[L]))
        out[L] = float(np.mean(sims)) if sims else float("nan")
    return out


def main():
    hs = torch.load(HS_PATH, map_location="cpu")
    meta = json.load(open(META_PATH))

    grp = {
        "emotional": keys_for(meta, EMOTIONAL),
        "matched_neutral": keys_for(meta, NEUTRAL),
        "positive": keys_for(meta, POSITIVE),
        "old_neutral": keys_for(meta, OLD_NEUTRAL),
    }
    print("Group sizes (sessions):", {k: len(v) for k, v in grp.items()}, "| baseline:", len(baseline_keys(meta)))
    if not grp["matched_neutral"]:
        print("\n⚠ No matched_neutral sessions found yet — extraction not finished. Re-run when done.")
        return

    rows = []
    # A) the narrativity control
    rows += probe_pair(hs, meta, grp["emotional"], grp["matched_neutral"], "emotional_vs_matched_neutral")
    # B) the arousal/valence control
    if grp["positive"]:
        rows += probe_pair(hs, meta, grp["emotional"], grp["positive"], "emotional_vs_positive")
        rows += probe_pair(hs, meta, grp["matched_neutral"], grp["positive"], "matched_neutral_vs_positive")

    # Per-layer distance-from-baseline (1 - cos) for each family
    print("\n── Distance from baseline (1 − cosine), per layer ──")
    dists = {name: cosine_to_baseline(hs, meta, keys, name) for name, keys in grp.items() if keys}
    hdr = "  layer  " + "".join(f"{n[:9]:>11}" for n in dists)
    print(hdr)
    for L in LAYERS:
        line = f"  {L:5d}  " + "".join(f"{1 - dists[n][L]:>11.4f}" for n in dists)
        print(line)

    # Key takeaway numbers at the emotion-specific band
    print("\n── Emotion-specific contrasts (distance gaps; positive = emotional rotates more) ──")
    for L in [40, 50, 60]:
        d = {n: 1 - dists[n][L] for n in dists}
        g_neu = d["emotional"] - d.get("matched_neutral", float("nan"))
        g_pos = d["emotional"] - d.get("positive", float("nan"))
        print(f"  layer {L}: emotional−matched_neutral = {g_neu:+.4f}   emotional−positive = {g_pos:+.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import csv
    if rows:
        with open(OUT_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nSaved probe results → {OUT_PATH}")

    # Data-driven verdict — print what the data ACTUALLY shows, not a generic key
    # (avoids a misleading "valence confound ruled out" line that contradicts the result).
    band = [40, 50, 60]
    gp = np.mean([(1 - dists["emotional"][L]) - (1 - dists["positive"][L]) for L in band]) if "positive" in dists else float("nan")
    gn = np.mean([(1 - dists["emotional"][L]) - (1 - dists["matched_neutral"][L]) for L in band]) if "matched_neutral" in dists else float("nan")
    print("\nVERDICT (middle-layer band 40–60):")
    print(f"  emotional − positive        = {gp:+.3f}   (≈0 ⇒ trauma & joy equidistant from baseline)")
    print(f"  emotional − matched_neutral = {gn:+.3f}")
    if abs(gp) < 0.03 and gn > 0.02:
        print("  ⇒ distance-from-baseline tracks AROUSAL, not valence: trauma ≈ joy, both > neutral.")
        print("    This is NOT an anxiety-specific representation. (The F1=1.0 probe rows are")
        print("    topic-driven and uninformative about valence — do not cite them as a valence signal.)")
        print("    For the valence DIRECTION test see valence_flip_analysis.py.")
    elif gp > 0.03:
        print("  ⇒ emotional rotates more than BOTH neutral and positive → candidate negative-valence signal.")
    else:
        print("  ⇒ emotional ≈ neutral → narrativity, no emotion-specific signal.")


if __name__ == "__main__":
    main()
