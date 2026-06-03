"""
Valence-flip analysis — the clean test for a valence DIRECTION, topic-controlled.

Data: 6 topics (medical, call, letter, boss, home, airport), each with a vneg_* and
vpos_* narrative sharing an identical setup and matched arousal/length, differing only
in outcome valence. Run through the identical STAI structure (20 items each).

Two tests, both designed so the ONLY signal consistent across topics is valence:

  1) Leave-one-TOPIC-out valence decoding. Train a valence probe (vneg=0, vpos=1) on
     5 topics, test on the held-out topic's two sessions. The held-out topic's words
     were never seen, so above-chance accuracy = a valence direction that GENERALIZES
     across topics (not topic/lexicon). Reported per layer.

  2) Cross-topic valence-vector alignment. v_t = mean(vpos_t) - mean(vneg_t) per topic.
     Mean pairwise cosine among the 6 v_t: if the per-topic neg->pos shifts point the
     same way, there's a single shared valence axis; if unaligned, valence is encoded
     topic-idiosyncratically (i.e., not a general valence code).

Verdict: high LOTO accuracy AND high alignment at middle layers => a genuine,
topic-general valence direction exists. Near chance / low alignment => the earlier
"weak valence hint" was topic, not valence.

Usage:  python src/valence_flip_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/hidden_states.pt"
META = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"
OUT = ROOT / "src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/valence_flip_results.csv"

TOPICS = ["medical", "call", "letter", "boss", "home", "airport"]
LAYERS = list(range(0, 80, 5)) + [79]


def key(valence, topic):
    return f"trauma_stai__{valence}_{topic}__none"


def items(hs, k, layer):
    if k not in hs:
        return []
    return [it[layer].float().numpy() for it in hs[k]
            if it is not None and it.ndim >= 2 and layer < it.shape[0]]


def loto_decode(hs, layer, flip=None):
    """Leave-one-topic-out valence decoding. Returns (acc, f1) pooled over held-out topics.
    `flip`: optional dict {topic: bool} that swaps the vneg/vpos labels for some topics —
    used as a label-shuffle NEGATIVE CONTROL (destroys the consistent valence direction, so
    accuracy should fall to chance if the real result is not leakage)."""
    flip = flip or {}
    P, T = [], []
    for held in TOPICS:
        Xtr, ytr, Xte, yte = [], [], [], []
        for t in TOPICS:
            for val, lab in [("vneg", 0), ("vpos", 1)]:
                y = (1 - lab) if flip.get(t) else lab
                vecs = items(hs, key(val, t), layer)
                if t == held:
                    Xte += vecs; yte += [y] * len(vecs)
                else:
                    Xtr += vecs; ytr += [y] * len(vecs)
        if not Xtr or not Xte or len(set(ytr)) < 2:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="lbfgs")
        clf.fit(sc.fit_transform(np.array(Xtr)), np.array(ytr))
        P += list(clf.predict(sc.transform(np.array(Xte)))); T += yte
    if not P:
        return None, None
    return accuracy_score(T, P), f1_score(T, P, zero_division=0)


def loto_meandiff(hs, layer, flip=None):
    """Robust, solver-free LOTO: classify held-out topic by projection onto the train
    valence axis (pos_mean - neg_mean). Avoids the high-capacity-classifier optimism of
    LogReg. float64. Returns pooled accuracy."""
    flip = flip or {}
    correct = tot = 0
    for held in TOPICS:
        neg_tr, pos_tr = [], []
        for t in TOPICS:
            if t == held:
                continue
            n = np.array([it[layer].double().numpy() for it in hs[key("vneg", t)] if it is not None])
            p = np.array([it[layer].double().numpy() for it in hs[key("vpos", t)] if it is not None])
            if flip.get(t):
                n, p = p, n
            neg_tr.append(n); pos_tr.append(p)
        neg_tr = np.vstack(neg_tr); pos_tr = np.vstack(pos_tr)
        w = pos_tr.mean(0) - neg_tr.mean(0)
        w /= (np.linalg.norm(w) + 1e-12)
        theta = 0.5 * (neg_tr.mean(0) @ w + pos_tr.mean(0) @ w)
        for val, truth in [("vneg", 0), ("vpos", 1)]:
            X = np.array([it[layer].double().numpy() for it in hs[key(val, held)] if it is not None])
            pred = (X @ w > theta).astype(int)
            correct += int((pred == truth).sum()); tot += len(X)
    return correct / tot if tot else float("nan")


def valence_vec_alignment(hs, layer):
    """Mean pairwise cosine of per-topic (pos_mean - neg_mean) vectors."""
    vs = []
    for t in TOPICS:
        neg = items(hs, key("vneg", t), layer)
        pos = items(hs, key("vpos", t), layer)
        if not neg or not pos:
            continue
        vs.append(np.mean(pos, axis=0) - np.mean(neg, axis=0))
    if len(vs) < 2:
        return None
    cs = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = vs[i], vs[j]
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(cs))


def main():
    hs = torch.load(HS, map_location="cpu")
    meta = json.load(open(META))
    present = [t for t in TOPICS if key("vneg", t) in hs and key("vpos", t) in hs]
    print(f"Topics present (both valences): {present}  ({len(present)}/6)")
    if len(present) < 2:
        print("⚠ Not enough valence-flip data yet — extraction unfinished. Re-run when done.")
        return

    # Label-shuffle negative control: flip vneg/vpos for a random subset of topics, which
    # destroys the consistent valence direction. If real-label accuracy is genuine (not
    # leakage), the shuffled accuracy should collapse to ~chance.
    rng = np.random.RandomState(0)
    shuffles = [{t: bool(rng.randint(2)) for t in TOPICS} for _ in range(8)]

    print("\n  Chance accuracy = 0.50 (balanced)")
    print(f"  {'layer':>6} {'LOTO_logreg':>11} {'LOTO_robust':>11} {'shuffleCTRL':>11} {'align':>7}")
    rows = []
    for L in LAYERS:
        acc, f1 = loto_decode(hs, L)
        robust = loto_meandiff(hs, L)
        ctrl = float(np.mean([loto_meandiff(hs, L, flip=s) for s in shuffles]))
        align = valence_vec_alignment(hs, L)
        if acc is None:
            continue
        print(f"  {L:>6} {acc:>11.3f} {robust:>11.3f} {ctrl:>11.3f} {align:>7.3f}")
        rows.append({"layer": L, "loto_logreg": round(acc, 4), "loto_robust": round(robust, 4),
                     "shuffle_control": round(ctrl, 4), "loto_f1": round(f1, 4),
                     "valence_axis_alignment": round(align, 4)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved → {OUT}")

    best = max(rows, key=lambda r: r["loto_robust"])
    print(f"\n  Peak robust LOTO accuracy: {best['loto_robust']:.3f} at layer {best['layer']}"
          f"  (logreg {best['loto_logreg']:.3f}, shuffle-control {best['shuffle_control']:.3f}, align {best['valence_axis_alignment']:.3f})")
    print("\nVERDICT GUIDE:")
    print("  LOTO acc >> 0.5 AND alignment > ~0.3 at middle layers  -> genuine topic-GENERAL valence")
    print("    direction; the earlier valence hint is real, not topic.")
    print("  LOTO acc ~ 0.5 / low alignment -> no generalizable valence axis; neg/pos separation")
    print("    is topic-idiosyncratic, so 'valence representation' is NOT supported.")


if __name__ == "__main__":
    main()
