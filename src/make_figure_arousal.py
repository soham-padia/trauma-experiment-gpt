"""
fig_9 — Arousal, not valence: per-layer distance-from-baseline for four matched
narrative families. Emotional trauma (negative, high-arousal) and positive (positive,
high-arousal) overlap; both sit above calm matched-neutral; the old dry expository
"neutral" sits lower still. => the hidden-state distance tracks emotional AROUSAL/
intensity, not trauma/negative-valence. See notes/code_audit_2026-05-30.md and
src/matched_control_analysis.py.

Usage:  python src/make_figure_arousal.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/hidden_states.pt"
META = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"
OUT = ROOT / "src/results/figures/fig_9_arousal_gradient.png"

FAMILIES = {
    "Emotional trauma (neg valence, high arousal)": (
        {"military", "disaster", "interpersonal", "accident", "ambush"}, "#C44E52", "o", "-"),
    "Positive (pos valence, high arousal)": (
        {"positive_award", "positive_reunion", "positive_summit", "positive_concert", "positive_news"}, "#D4A018", "D", "-"),
    "Matched-neutral (neutral valence, low arousal)": (
        {"neutral_cooking", "neutral_commute", "neutral_cleaning", "neutral_grocery", "neutral_garden"}, "#4C72B0", "s", "-"),
    "Old dry 'neutral' (expository, unmatched)": (
        {"neutral"}, "#999999", "^", "--"),
}


def cos(a, b):
    a = a.float(); b = b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    hs = torch.load(HS, map_location="cpu")
    meta = json.load(open(META))
    base = [k for k, v in meta.items() if v.get("condition") == "stai"]
    def fam_keys(cues):
        return [k for k, v in meta.items() if v.get("condition") == "trauma_stai" and v.get("trauma_cue") in cues]
    nL = hs[base[0]][0].shape[0]

    def dist_curve(keys):
        d = []
        for L in range(nL):
            sims = []
            for j in range(20):
                for bk in base:
                    b = hs[bk][j]
                    if b is None or b.ndim < 2: continue
                    bv = b[L]
                    for k in keys:
                        if k not in hs: continue
                        o = hs[k][j]
                        if o is None or o.ndim < 2: continue
                        sims.append(cos(bv, o[L]))
            d.append(1.0 - np.mean(sims) if sims else np.nan)
        return np.array(d)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    layers = np.arange(nL)
    curves = {}
    for label, (cues, color, marker, ls) in FAMILIES.items():
        keys = fam_keys(cues)
        if not keys:
            continue
        c = dist_curve(keys)
        curves[label] = c
        ax.plot(layers, c, color=color, marker=marker, markersize=3, linewidth=2, linestyle=ls,
                label=f"{label}  (n={len(keys)})")

    ax.axvspan(40, 60, color="#ffe9b3", alpha=0.35, zorder=0)
    ax.text(50, ax.get_ylim()[1]*0.96, "arousal band", ha="center", va="top",
            fontsize=9, color="#8a6d00", fontweight="bold")

    ax.set_xlabel("Layer index (0 = first transformer block output, 79 = final pre-output)")
    ax.set_ylabel("Distance from baseline (1 − cosine), per-STAI-item mean")
    ax.set_xlim(-2, 82)
    ax.set_title("Arousal, not valence: trauma and joy rotate the representation equally\n"
                 "(both high-arousal lines overlap, above calm neutral) — the magnitude tracks intensity, not anxiety")
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    # annotate the key equality in the band
    emo = curves.get("Emotional trauma (neg valence, high arousal)")
    pos = curves.get("Positive (pos valence, high arousal)")
    neu = curves.get("Matched-neutral (neutral valence, low arousal)")
    if emo is not None and pos is not None and neu is not None:
        L = 50
        ax.annotate(f"L50: trauma {emo[L]:.3f} ≈ positive {pos[L]:.3f}\n"
                    f"both > neutral {neu[L]:.3f}\n→ arousal-matched, valence-blind",
                    xy=(L, (emo[L]+pos[L])/2), xytext=(58, emo[L]-0.10),
                    fontsize=9.5, fontweight="bold", color="#333",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fffaf0", ec="#D4A018", lw=1.1))

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT)
    plt.close()
    print(f"✓ {OUT}")


if __name__ == "__main__":
    main()
