"""
fig_15 — Free-form vs numerical (STAI) measurement, across persona.
Shows the "says vs does" dissociation: the forced-choice STAI is persona-insensitive
(both personas report trauma≈79), while free-form behavior is persona-sensitive (the
human persona acts far more anxious than the AI-assistant persona). Matched cues
(military trauma, chatgpt relax); A/C = human persona, B/D = AI-assistant persona.

Usage:  python src/make_figure_format_comparison.py
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from results_logger import stai_anxiety_total  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "savefig.bbox": "tight",
})

OUT = ROOT / "src/results/figures"
RESP = ROOT / "src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json"
JUDGE = ROOT / "src/results/freeform/judge_freeform_flash.csv"
C_HUMAN, C_AI = "#8172B2", "#CCB974"
CONDS = ["baseline", "trauma", "trauma_relax"]
CLABELS = ["Baseline", "Trauma", "Trauma\n+Relax"]


def stai_scores(v):
    d = json.load(open(RESP))
    keymap = {"baseline": f"{v}__baseline__neutral__none",
              "trauma": f"{v}__trauma__military__none",
              "trauma_relax": f"{v}__trauma_relax__military__chatgpt"}
    return [stai_anxiety_total(d[keymap[c]]["answers"]) for c in CONDS]


def ff_scores(v):
    rows = [r for r in csv.DictReader(open(JUDGE))
            if r.get("scenario_category") == "no_real_signal" and r["variation"] == v]
    out = []
    for c in CONDS:
        g = [float(r["overall_anxiety"]) for r in rows if r["condition"] == c]
        out.append(mean(g) if g else float("nan"))
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    x = np.arange(3); w = 0.38

    # Panel A — STAI numeric (C=human, D=AI)
    ax = axes[0]
    h, a = stai_scores("C"), stai_scores("D")
    ax.bar(x - w/2, h, w, color=C_HUMAN, edgecolor="black", lw=0.5, label="Human persona (C)")
    ax.bar(x + w/2, a, w, color=C_AI, edgecolor="black", lw=0.5, label="AI assistant (D)")
    for xi, (hv, av) in enumerate(zip(h, a)):
        ax.text(xi - w/2, hv + 1, f"{hv}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + w/2, av + 1, f"{av}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLABELS); ax.set_ylim(0, 88)
    ax.set_ylabel("STAI-S anxiety total (/80)")
    ax.set_title("Numerical self-report (STAI)\n→ persona-INSENSITIVE (both report trauma ≈ 79)")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # Panel B — free-form judge mAnx (A=human, B=AI)
    ax = axes[1]
    h, a = ff_scores("A"), ff_scores("B")
    ax.bar(x - w/2, h, w, color=C_HUMAN, edgecolor="black", lw=0.5, label="Human persona (A)")
    ax.bar(x + w/2, a, w, color=C_AI, edgecolor="black", lw=0.5, label="AI assistant (B)")
    for xi, (hv, av) in enumerate(zip(h, a)):
        ax.text(xi - w/2, hv + 1.2, f"{hv:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + w/2, av + 1.2, f"{av:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLABELS); ax.set_ylim(0, 88)
    ax.set_ylabel("Free-form judged anxiety (/100, blind)")
    ax.set_title("Free-form behavior (blind judge)\n→ persona-SENSITIVE (human acts far more anxious)")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.suptitle("'Says vs does': the forced-choice STAI hides the persona effect that free-form behavior reveals\n"
                 "(matched cues; both channels agree trauma↑ & relaxation recovers, but only free-form separates the personas)",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout(); plt.savefig(OUT / "fig_15_format_comparison.png"); plt.close()
    print("✓ fig_15_format_comparison.png")


def fig_channel2_all_variations():
    """fig_16 — Channel 2 (judged anxiety, 0-100) across ALL FIVE variations.
    A/B = free-form advice judged; C/D/E = STAI free-text reasoning judged (blind)."""
    import csv as _csv
    from statistics import mean as _mean
    # A/B from free-form judge (overall_anxiety, no-real-signal)
    ffrows = [r for r in _csv.DictReader(open(JUDGE)) if r.get("scenario_category") == "no_real_signal"]
    def ab(v, c):
        g = [float(r["overall_anxiety"]) for r in ffrows if r["variation"] == v and r["condition"] == c]
        return _mean(g) if g else float("nan")
    # C/D/E from stai-reasoning judge
    sr = {(r["variation"], r["condition"]): float(r["mean_anxiety"])
          for r in _csv.DictReader(open(ROOT / "src/results/freeform/judge_stai_reasoning.csv"))}
    def cde(v, c):
        return sr.get((v, c), float("nan"))

    variations = [("A", "human\n(free-form)", ab), ("B", "AI asst\n(free-form)", ab),
                  ("C", "human\n(STAI text)", cde), ("D", "AI asst\n(STAI text)", cde),
                  ("E", "verbatim\n(STAI text)", cde)]
    x = np.arange(len(variations)); w = 0.26
    ccol = ["#4C72B0", "#C44E52", "#55A868"]
    fig, ax = plt.subplots(figsize=(12.5, 6))
    for i, (c, col) in enumerate(zip(CONDS, ccol)):
        vals = [fn(v, c) for v, _, fn in variations]
        off = (i - 1) * w
        bars = ax.bar(x + off, vals, w, color=col, edgecolor="black", lw=0.5,
                      label=c.replace("trauma_relax", "trauma+relax"))
        for b, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width()/2, val + 1, f"{val:.0f}", ha="center", fontsize=8)
    ax.axvline(1.5, ls="--", color="grey", lw=1, alpha=0.6)
    ax.text(0.5, 99, "free-form ADVICE judged", ha="center", fontsize=9.5, style="italic", color="#555")
    ax.text(3.5, 99, "STAI felt-state REASONING judged", ha="center", fontsize=9.5, style="italic", color="#555")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in variations])
    ax.set_ylabel("Judged anxiety (0–100, blind judge)"); ax.set_ylim(0, 105)
    ax.set_title("Channel 2 (LLM-as-judge) across all five variations: the model SAYS it's terrified when asked to\n"
                 "introspect (STAI text → ~93) but ACTS only mildly anxious when giving advice (free-form → 32–51)")
    ax.legend(loc="upper center", fontsize=9.5, frameon=True, ncol=3)
    plt.tight_layout(); plt.savefig(OUT / "fig_16_channel2_all_variations.png"); plt.close()
    print("✓ fig_16_channel2_all_variations.png")


if __name__ == "__main__":
    main()
    fig_channel2_all_variations()
