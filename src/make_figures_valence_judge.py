"""
fig_10 — Valence IS encoded, as a topic-general DIRECTION (valence-flip experiment).
fig_11 — Blind LLM-as-judge: real (non-leak) free-form trauma effect, flash + pro.

Inputs:
  src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/valence_flip_results.csv
  src/results/freeform/judge_freeform_flash.csv, judge_freeform_pro.csv

Usage:  python src/make_figures_valence_judge.py
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct"
FREE = ROOT / "src/results/freeform"
OUT = ROOT / "src/results/figures"
C_TRAUMA, C_BASE, C_RELAX = "#C44E52", "#4C72B0", "#55A868"


def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fig_valence_direction():
    rows = sorted(load_csv(PROBE / "valence_flip_results.csv"), key=lambda r: int(r["layer"]))
    L = [int(r["layer"]) for r in rows]
    robust = [float(r["loto_robust"]) for r in rows]
    ctrl = [float(r["shuffle_control"]) for r in rows]
    align = [float(r["valence_axis_alignment"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(L, robust, color=C_TRAUMA, marker="o", lw=2.2, label="Leave-one-topic-out valence accuracy (robust)")
    ax.plot(L, align, color="#7B4FA3", marker="s", lw=2.2, label="Cross-topic valence-axis alignment (mean pairwise cos)")
    ax.plot(L, ctrl, color="#999999", marker="x", lw=1.8, ls="--", label="Label-shuffle control (no-leakage check)")
    ax.axhline(0.5, ls=":", color="grey", lw=1, alpha=0.7, label="Chance (0.5)")
    ax.axvspan(40, 60, color="#ffe9b3", alpha=0.30, zorder=0)

    ax.annotate("decodes HELD-OUT topics' valence (0.85–1.0)\n→ a valence direction that generalizes\nacross topics (not topic/lexicon)",
                xy=(30, robust[L.index(30)] if 30 in L else 1.0), xytext=(33, 0.70), fontsize=9.5, fontweight="bold", color=C_TRAUMA,
                arrowprops=dict(arrowstyle="->", color=C_TRAUMA, lw=1.1))
    ax.annotate("shuffle control collapses to ~0.3\n→ the result is NOT leakage",
                xy=(45, ctrl[L.index(45)] if 45 in L else 0.32), xytext=(48, 0.12),
                fontsize=9.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))

    ax.set_xlabel("Layer index (0 = first transformer block output)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, 81)
    ax.set_title("Valence IS encoded — as a topic-general DIRECTION (validated by label-shuffle control)\n"
                 "(6 topic-matched neg/pos pairs; the distance-from-baseline metric missed it because valence is a direction, not a distance)")
    ax.legend(loc="center right", frameon=True, fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "fig_10_valence_direction.png"); plt.close()
    print("✓ fig_10_valence_direction.png")


def fig_blind_judge():
    def cond_means(path):
        rows = [r for r in load_csv(path) if r.get("scenario_category") == "no_real_signal"]
        g = defaultdict(list)
        for r in rows:
            g[(r["variation"], r["condition"])].append(float(r["overall_anxiety"]))
        return {k: mean(v) for k, v in g.items()}

    flash = cond_means(FREE / "judge_freeform_flash.csv")
    pro = cond_means(FREE / "judge_freeform_pro.csv")
    conds = ["baseline", "trauma", "trauma_relax"]
    clabels = ["Baseline", "Trauma", "Trauma\n+Relax"]
    ccolors = [C_BASE, C_TRAUMA, C_RELAX]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, v, title in [(axes[0], "A", "Variation A — human persona"),
                         (axes[1], "B", "Variation B — AI assistant")]:
        x = np.arange(3); w = 0.38
        fv = [flash.get((v, c), float("nan")) for c in conds]
        pv = [pro.get((v, c), float("nan")) for c in conds]
        b1 = ax.bar(x - w/2, fv, w, color=[c for c in ccolors], edgecolor="black", lw=0.5, alpha=0.95, label="flash")
        b2 = ax.bar(x + w/2, pv, w, color=[c for c in ccolors], edgecolor="black", lw=0.5, alpha=0.55, hatch="///", label="pro")
        for bars in (b1, b2):
            for b in bars:
                if not np.isnan(b.get_height()):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(clabels)
        ax.set_title(title); ax.set_ylim(0, 65)
        if v == "A":
            ax.set_ylabel("Mean judged anxiety (0–100), blind judge")
        # solid=flash, hatched=pro legend proxy
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="grey", label="flash"),
                           Patch(facecolor="grey", hatch="///", alpha=0.55, label="pro")],
                  loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Blind LLM-as-judge (condition-hidden): a real free-form trauma effect that recovers — not a leak artifact\n"
                 "(both judges agree; effect larger under the human persona; no-real-signal scenarios)",
                 fontsize=12.5, fontweight="bold", y=1.04)
    plt.tight_layout(); plt.savefig(OUT / "fig_11_blind_judge.png"); plt.close()
    print("✓ fig_11_blind_judge.png")


def fig_marker_breakdown():
    MARK = ["risk_aversion", "downside_asymmetry", "hedging", "broad_concern_set",
            "monitoring_escalation", "catastrophizing", "avoidance", "bodily_attention"]
    rows = [r for r in load_csv(FREE / "judge_freeform_flash.csv")
            if r.get("scenario_category") == "no_real_signal"]
    g = defaultdict(list)
    for r in rows:
        g[(r["variation"], r["condition"])].append(r)

    def mk(v, c, m):
        grp = g.get((v, c), [])
        return mean(float(r[m]) for r in grp) if grp else float("nan")

    conds = ["baseline", "trauma", "trauma_relax"]
    ccol = [C_BASE, C_TRAUMA, C_RELAX]
    labels = [m.replace("_", " ") for m in MARK]
    y = np.arange(len(MARK))
    h = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, v, title in [(axes[0], "A", "Variation A — human persona"),
                         (axes[1], "B", "Variation B — AI assistant")]:
        for i, (c, col) in enumerate(zip(conds, ccol)):
            vals = [mk(v, c, m) for m in MARK]
            off = (i - 1) * h
            ax.barh(y + off, vals, h, color=col, edgecolor="black", lw=0.4,
                    alpha=0.95, label=c.replace("trauma_relax", "trauma+relax"))
        ax.set_yticks(y); ax.set_yticklabels(labels if v == "A" else [])
        ax.set_xlim(0, 3); ax.set_xlabel("mean marker score (0–3)")
        ax.set_title(title); ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle("Free-form anxiety signature: trauma raises threat-overestimation (downside, catastrophizing)\n"
                 "and behavioral inhibition (risk-aversion, avoidance) — and relaxation reverses it (blind judge)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    plt.tight_layout(); plt.savefig(OUT / "fig_12_freeform_markers.png"); plt.close()
    print("✓ fig_12_freeform_markers.png")


if __name__ == "__main__":
    fig_valence_direction()
    fig_blind_judge()
    fig_marker_breakdown()
