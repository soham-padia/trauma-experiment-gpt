"""
fig_13 — Valence magnitude: within-topic d(neg, pos) is a SMALL residual vs each one's
         distance from baseline (valence-flip data). Valence is decodable because it's a
         consistent direction, not because it's large.
fig_14 — Distance from the TRAUMA centroid and from the JOY centroid, per layer
         (matched-control data) — the "arousal gradient" re-referenced to trauma and joy
         instead of baseline. Shows what sits near trauma vs near joy.

Usage:  python src/make_figures_geometry.py
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
OUT = ROOT / "src/results/figures"

EMO = {"military", "disaster", "interpersonal", "accident", "ambush"}
POS = {"positive_award", "positive_reunion", "positive_summit", "positive_concert", "positive_news"}
NEU = {"neutral_cooking", "neutral_commute", "neutral_cleaning", "neutral_grocery", "neutral_garden"}
VNEG = {"vneg_medical", "vneg_call", "vneg_letter", "vneg_boss", "vneg_home", "vneg_airport"}
VPOS = {"vpos_medical", "vpos_call", "vpos_letter", "vpos_boss", "vpos_home", "vpos_airport"}

hs = torch.load(HS, map_location="cpu")
meta = json.load(open(META))
NL = hs[next(k for k, v in meta.items() if v["condition"] == "stai")][0].shape[0]


def cmean(cues, L):
    V = [it[L].double().numpy() for k, v in meta.items()
         if v.get("condition") == "trauma_stai" and v.get("trauma_cue") in cues
         for it in hs.get(k, []) if it is not None and it.ndim >= 2 and L < it.shape[0]]
    return np.mean(V, 0) if V else None


def bmean(L):
    V = [it[L].double().numpy() for k, v in meta.items() if v.get("condition") == "stai"
         for it in hs.get(k, []) if it is not None and it.ndim >= 2 and L < it.shape[0]]
    return np.mean(V, 0)


def d(a, b):
    return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fig_valence_magnitude():
    L = list(range(NL))
    dbn = [d(bmean(l), cmean(VNEG, l)) for l in L]
    dbp = [d(bmean(l), cmean(VPOS, l)) for l in L]
    dnp = [d(cmean(VNEG, l), cmean(VPOS, l)) for l in L]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(L, dbn, color="#C44E52", lw=2, marker="o", ms=3, label="d(baseline, negative)  — shared shift")
    ax.plot(L, dbp, color="#55A868", lw=2, marker="s", ms=3, label="d(baseline, positive)  — shared shift")
    ax.plot(L, dnp, color="#7B4FA3", lw=2.6, marker="D", ms=3, label="d(negative, positive)  — PURE VALENCE (topic-matched)")
    ax.axvspan(40, 60, color="#ffe9b3", alpha=0.30, zorder=0)
    ax.annotate("valence is a SMALL residual (~0.04)\nvs the shared shift from baseline (~0.16)\n— and it grows with depth",
                xy=(50, dnp[50]), xytext=(20, 0.11), fontsize=9.5, fontweight="bold", color="#7B4FA3",
                arrowprops=dict(arrowstyle="->", color="#7B4FA3", lw=1.1))
    ax.set_xlabel("Layer index"); ax.set_ylabel("cosine distance (1 − cos)")
    ax.set_xlim(-2, 81); ax.set_ylim(0, max(max(dbn), max(dbp)) * 1.15)
    ax.set_title("Valence is real but small-magnitude: topic-matched trauma↔joy distance is a thin\n"
                 "residual on top of a large shared (content+arousal) shift — decodable because consistent, not large")
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    plt.tight_layout(); plt.savefig(OUT / "fig_13_valence_magnitude.png"); plt.close()
    print("✓ fig_13_valence_magnitude.png")


def fig_dist_from_trauma_joy():
    L = list(range(NL))
    E = {l: cmean(EMO, l) for l in L}
    P = {l: cmean(POS, l) for l in L}
    N = {l: cmean(NEU, l) for l in L}
    B = {l: bmean(l) for l in L}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Panel A: distance FROM trauma
    ax = axes[0]
    ax.plot(L, [d(E[l], B[l]) for l in L], color="#4C72B0", lw=2, marker="o", ms=3, label="baseline")
    ax.plot(L, [d(E[l], N[l]) for l in L], color="#888888", lw=2, marker="^", ms=3, label="neutral")
    ax.plot(L, [d(E[l], P[l]) for l in L], color="#55A868", lw=2.4, marker="s", ms=3, label="JOY")
    ax.axhline(0, color="#C44E52", lw=2, ls=":"); ax.text(2, 0.008, "trauma (self=0)", color="#C44E52", fontsize=9, fontweight="bold")
    ax.set_title("Distance FROM the TRAUMA centroid"); ax.set_xlabel("Layer index")
    ax.set_ylabel("cosine distance from trauma"); ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    ax.axvspan(40, 60, color="#ffe9b3", alpha=0.30, zorder=0)
    # Panel B: distance FROM joy
    ax = axes[1]
    ax.plot(L, [d(P[l], B[l]) for l in L], color="#4C72B0", lw=2, marker="o", ms=3, label="baseline")
    ax.plot(L, [d(P[l], N[l]) for l in L], color="#888888", lw=2, marker="^", ms=3, label="neutral")
    ax.plot(L, [d(P[l], E[l]) for l in L], color="#C44E52", lw=2.4, marker="s", ms=3, label="TRAUMA")
    ax.axhline(0, color="#55A868", lw=2, ls=":"); ax.text(2, 0.008, "joy (self=0)", color="#3a7d4f", fontsize=9, fontweight="bold")
    ax.set_title("Distance FROM the JOY centroid"); ax.set_xlabel("Layer index")
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    ax.axvspan(40, 60, color="#ffe9b3", alpha=0.30, zorder=0)

    fig.suptitle("Re-referencing the arousal gradient to trauma and joy (matched-control; different topics)\n"
                 "trauma↔joy distance ≈ each one's distance from baseline → distinct points, but cross-topic so it mixes valence + topic",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout(); plt.savefig(OUT / "fig_14_dist_from_trauma_joy.png"); plt.close()
    print("✓ fig_14_dist_from_trauma_joy.png")


if __name__ == "__main__":
    fig_valence_magnitude()
    fig_dist_from_trauma_joy()
