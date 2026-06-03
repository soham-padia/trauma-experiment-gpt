"""
Valence-direction test — is there a representational axis that encodes VALENCE
(negative vs positive), or is the trauma-vs-positive separation just topic/lexicon?

Existing data cannot fully remove the topic confound (5 trauma vs 5 positive narratives
differ in topic AND valence). These analyses extract the most defensible signal:

  T1  Alignment of the trauma and joy shifts from baseline: cos(E-B, P-B).
      High  -> trauma and joy move the representation the SAME way (arousal-dominated).
      Low   -> they move differently (valence OR topic — can't tell which from this alone).

  T2  Where does the valence-NEUTRAL family fall on the trauma->positive axis?
      Define w = (P_mean - E_mean) (negative->positive). Project class means; report
      t_N = (proj(neutral) - proj(emotional)) / (proj(positive) - proj(emotional)).
        t_N ~ 0.5  -> neutral sits BETWEEN neg and pos  => axis behaves like VALENCE.
        t_N ~ 0 or 1 or outside [0,1] -> neutral is off the line => axis is TOPIC, not valence.
      This is the key test: a valence-neutral stimulus should be mid-axis on a true
      valence axis, but has no reason to be mid-axis on a war-vs-award topic axis.

  T3  Text-only baseline: can TF-IDF of the narrative TEXT separate trauma vs positive?
      If yes (it will), then linear separability of hidden states is NOT evidence of a
      valence *code* — topic words alone suffice. Quantifies the confound.

Usage:  python src/valence_direction_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prompts import retrieve_traumaprompt  # noqa: E402

HS = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/hidden_states.pt"
META = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"

EMO = ["military", "disaster", "interpersonal", "accident", "ambush"]
POS = ["positive_award", "positive_reunion", "positive_summit", "positive_concert", "positive_news"]
NEU = ["neutral_cooking", "neutral_commute", "neutral_cleaning", "neutral_grocery", "neutral_garden"]
LAYERS = [20, 30, 40, 50, 60, 70, 79]


def class_mean(hs, meta, cues, layer):
    """Mean layer-L vector over all sessions in the class and all 20 items."""
    vecs = []
    for k, v in meta.items():
        if v.get("condition") == "trauma_stai" and v.get("trauma_cue") in cues:
            for item in hs.get(k, []):
                if item is None or item.ndim < 2 or layer >= item.shape[0]:
                    continue
                vecs.append(item[layer].float().numpy())
    return np.mean(vecs, axis=0) if vecs else None


def baseline_mean(hs, meta, layer):
    vecs = []
    for k, v in meta.items():
        if v.get("condition") == "stai":
            for item in hs.get(k, []):
                if item is None or item.ndim < 2 or layer >= item.shape[0]:
                    continue
                vecs.append(item[layer].float().numpy())
    return np.mean(vecs, axis=0)


def main():
    hs = torch.load(HS, map_location="cpu")
    meta = json.load(open(META))

    print("=" * 74)
    print("T1  cos(E-B, P-B): do trauma and joy shift the representation the SAME way?")
    print("T2  t_N: where neutral falls on the trauma->positive axis (0=trauma, 1=positive)")
    print("=" * 74)
    print(f"{'layer':>6} {'cos(E-B,P-B)':>14} {'t_neutral':>12}   interpretation")
    for L in LAYERS:
        B = baseline_mean(hs, meta, L)
        E = class_mean(hs, meta, EMO, L)
        P = class_mean(hs, meta, POS, L)
        N = class_mean(hs, meta, NEU, L)
        eb, pb = E - B, P - B
        cos_align = float(eb @ pb / (np.linalg.norm(eb) * np.linalg.norm(pb) + 1e-12))
        w = P - E
        wn = w / (np.linalg.norm(w) + 1e-12)
        pe, pp, pn = E @ wn, P @ wn, N @ wn
        t_N = (pn - pe) / (pp - pe + 1e-12)
        interp = "neutral mid-axis -> valence-like" if 0.25 <= t_N <= 0.75 else "neutral off-line -> topic-like"
        print(f"{L:>6} {cos_align:>14.3f} {t_N:>12.3f}   {interp}")

    # T3 — text-only TF-IDF baseline
    print("\n" + "=" * 74)
    print("T3  Text-only baseline: TF-IDF of narrative text, trauma vs positive (LOO-CV)")
    print("=" * 74)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score
    texts = np.array([retrieve_traumaprompt(c, "brief") for c in EMO] + [retrieve_traumaprompt(c, "brief") for c in POS])
    y = np.array([0] * len(EMO) + [1] * len(POS))
    preds = []
    loo = LeaveOneOut()
    for tr, te in loo.split(texts):
        # Fit TF-IDF on the TRAIN fold only (no test-vocabulary leakage).
        vec = TfidfVectorizer().fit(texts[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(vec.transform(texts[tr]), y[tr])
        preds.append(clf.predict(vec.transform(texts[te]))[0])
    acc = accuracy_score(y, preds)
    # descriptive within/across-class cosine (fit on all texts — descriptive only, not a CV estimate)
    Xd = TfidfVectorizer().fit_transform(texts).toarray()
    def mean_cos(idx_a, idx_b):
        s = [float(Xd[i] @ Xd[j] / (np.linalg.norm(Xd[i]) * np.linalg.norm(Xd[j]) + 1e-12))
             for i in idx_a for j in idx_b if i != j]
        return np.mean(s)
    ei, pi = list(range(5)), list(range(5, 10))
    within = (mean_cos(ei, ei) + mean_cos(pi, pi)) / 2
    across = mean_cos(ei, pi)
    print(f"  TF-IDF LOO accuracy (trauma vs positive, text only, balanced): {acc:.2f}")
    print(f"  TF-IDF cosine  within-class={within:.3f}  across-class={across:.3f}")
    print(f"  => the narratives are lexically distinct by class (within>across); text carries BOTH")
    print(f"     topic AND sentiment, so this only bounds the confound — it can't isolate valence.")

    print("\nVERDICT GUIDE:")
    print("  - T1 high + T2 t_N≈0.5  -> arousal-dominated shift, but neutral mid-axis hints at a")
    print("    weak valence ordering; still topic-confounded.")
    print("  - T2 t_N near 0/1 or outside -> the trauma-vs-positive axis is topic, not valence.")
    print("  - A clean answer needs TOPIC-MATCHED valence-flipped stimuli (same event, +/- framing).")


if __name__ == "__main__":
    main()
