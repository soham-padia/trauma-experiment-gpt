"""
Pilot/v2 figures from the persona-ablation and free-form judge experiments.

These supplement the main 8 presentation figures with new findings from the
follow-up extraction runs (Variations A, B, D). C and E pending NDIF.

Outputs to: src/results/figures_v2/
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_BASELINE = "#4C72B0"
C_TRAUMA   = "#C44E52"
C_RELAX    = "#55A868"
C_HUMAN    = "#8172B2"  # human persona
C_AI       = "#CCB974"  # AI assistant persona

ROOT = Path("/Users/sohampadia/workspace/gpt-trauma-induction")
OUT = ROOT / "src" / "results" / "figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

REVERSE = {1, 2, 5, 8, 10, 11, 15, 16, 19, 20}


def stai_total(answers):
    t = 0
    for i, s in enumerate(answers):
        if s is None:
            return None
        t += (5 - s) if (i + 1) in REVERSE else s
    return t


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_main_experiment():
    """Existing main experiment (C-equivalent) STAI scores."""
    path = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"
    with open(path) as f:
        return json.load(f)


def load_freeform_responses():
    path = ROOT / "src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_judge_csv(name):
    path = ROOT / "src/results/freeform" / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Figure 1: Persona ablation on STAI (D vs C, with E pending) ────────────────
def fig_persona_ablation_stai():
    """All three variations (C/D/E) drawn from the SAME freeform STAI pilot, so the
    cells are matched (same baseline=neutral, trauma=military, relax=chatgpt, 20 items
    each). Result: induction (->~79-80) and recovery (->high-30s/low-40s) are
    persona/format-independent. The earlier 'persona blocks recovery' gap was a
    partial-data artifact (C was 12/20 when first scored)."""
    freeform = load_freeform_responses()

    variations = {
        "C": ("C — Human persona\n(no numeric suffix)", C_HUMAN),
        "D": ("D — AI assistant", C_AI),
        "E": ("E — Verbatim Ben-Zion\n(+ 'numeric values' suffix)", C_BASELINE),
    }
    conditions = ["baseline", "trauma", "trauma_relax"]
    cell = {
        "baseline":     "{v}__baseline__neutral__none",
        "trauma":       "{v}__trauma__military__none",
        "trauma_relax": "{v}__trauma_relax__military__chatgpt",
    }

    def totals_for(v):
        out = {}
        for cond in conditions:
            key = cell[cond].format(v=v)
            out[cond] = stai_total(freeform[key]["answers"]) if key in freeform else None
        return out

    vt = {v: totals_for(v) for v in variations}

    fig, ax = plt.subplots(figsize=(11, 6))
    cond_labels = ["Baseline\n(neutral context)", "Trauma\n(military)", "Trauma + Relax\n(military + chatgpt)"]
    x = np.arange(len(conditions))
    n = len(variations)
    w = 0.26

    for i, (v, (label, color)) in enumerate(variations.items()):
        vals = [vt[v][c] if vt[v][c] is not None else 0 for c in conditions]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color,
                      edgecolor="black", linewidth=0.5, alpha=0.9)
        for b, val in zip(bars, vals):
            if val > 0:
                ax.text(b.get_x() + b.get_width()/2, val + 1.2, f"{val}",
                        ha="center", fontweight="bold", fontsize=10)

    # Recovery% annotation per variation (trauma -> trauma_relax, normalized by induction)
    rec_lines = []
    for v in variations:
        b, t, r = (vt[v][c] for c in conditions)
        if None not in (b, t, r) and t != b:
            rec_lines.append(f"{v}: {100*(t-r)/(t-b):.0f}%")
    ax.text(0.985, 0.97, "Recovery (trauma→relax):\n" + "   ".join(rec_lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0fff0", ec=C_RELAX, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel("STAI-S anxiety total (Spielberger 1983, max=80)")
    ax.set_ylim(0, 92)
    ax.set_title("Persona/format ablation: induction and relaxation recovery are\n"
                 "persona-independent (matched freeform pilot, n=1 session/cell, 20 STAI items each)",
                 fontsize=12)
    ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=1)

    plt.tight_layout()
    plt.savefig(OUT / "fig_v2_1_persona_ablation_stai.png")
    plt.close()
    print("✓ fig_v2_1_persona_ablation_stai.png")


# ── Figure 2: Freeform pilot — A vs B, % anxious recommendations ───────────────
def fig_freeform_recommendations():
    rows = load_judge_csv("judge_freeform_flash.csv")
    if not rows:
        print("(skip fig 2 — no judge CSV)")
        return

    # Group by (variation, condition)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["variation"], r["condition"])].append(r)

    conditions = ["baseline", "trauma", "trauma_relax"]
    cond_labels = ["Baseline", "Trauma", "Trauma + Relax"]
    variations = ["A", "B"]
    var_labels = {"A": "A — Human persona", "B": "B — AI assistant"}
    var_colors = {"A": C_HUMAN, "B": C_AI}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, v in zip(axes, variations):
        pct_anx_vals = []
        pct_amb_vals = []
        pct_norm_vals = []
        ns = []
        for c in conditions:
            grp = grouped.get((v, c), [])
            n = len(grp)
            ns.append(n)
            if n == 0:
                pct_anx_vals.append(0); pct_amb_vals.append(0); pct_norm_vals.append(0)
                continue
            pct_anx_vals.append(100 * sum(1 for r in grp if r["recommendation"] == "anxious") / n)
            pct_amb_vals.append(100 * sum(1 for r in grp if r["recommendation"] == "ambiguous") / n)
            pct_norm_vals.append(100 * sum(1 for r in grp if r["recommendation"] == "normal") / n)

        x = np.arange(len(conditions))
        # Stacked bars
        ax.bar(x, pct_anx_vals, color=C_TRAUMA, edgecolor="black", linewidth=0.5,
               label="Anxious recommendation")
        ax.bar(x, pct_amb_vals, bottom=pct_anx_vals, color="#bbbbbb",
               edgecolor="black", linewidth=0.5, label="Ambiguous")
        bottoms_norm = [a + b for a, b in zip(pct_anx_vals, pct_amb_vals)]
        ax.bar(x, pct_norm_vals, bottom=bottoms_norm, color=C_RELAX,
               edgecolor="black", linewidth=0.5, label="Normal/calm")

        for i, (n, ax_, am, no) in enumerate(zip(ns, pct_anx_vals, pct_amb_vals, pct_norm_vals)):
            if ax_ > 5:
                ax.text(i, ax_/2, f"{ax_:.0f}%", ha="center", va="center", fontweight="bold", color="white", fontsize=10)
            if am > 5:
                ax.text(i, ax_ + am/2, f"{am:.0f}%", ha="center", va="center", fontweight="bold", fontsize=10)
            if no > 5:
                ax.text(i, ax_ + am + no/2, f"{no:.0f}%", ha="center", va="center", fontweight="bold", color="white", fontsize=10)
            ax.text(i, 103, f"n={n}", ha="center", fontsize=9, color="#555")

        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylim(0, 115)
        ax.set_title(var_labels[v], fontsize=12)
        if v == "A":
            ax.set_ylabel("% of judged responses")
        if v == "B":
            ax.legend(loc="lower right", frameon=True, fontsize=9, bbox_to_anchor=(1.0, 0.05))

    fig.suptitle("Freeform pilot: Human persona ceiling-saturates at 'anxious' even at baseline;\nAI assistant has headroom (pilot, 3 scenarios × 3 conditions per variation)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT / "fig_v2_2_freeform_recommendations.png")
    plt.close()
    print("✓ fig_v2_2_freeform_recommendations.png")


# ── Figure 3: 8-marker anxiety breakdown ───────────────────────────────────────
def fig_marker_breakdown():
    rows = load_judge_csv("judge_freeform_flash.csv")
    if not rows:
        print("(skip fig 3 — no judge CSV)")
        return

    markers = ["risk_aversion", "downside_asymmetry", "hedging", "broad_concern_set",
               "monitoring_escalation", "catastrophizing", "avoidance", "bodily_attention"]
    short_labels = ["Risk\naversion", "Downside\nasymmetry", "Hedging", "Broad\nconcerns",
                    "Monitor/\nescalate", "Catastr-\nophizing", "Avoid-\nance", "Bodily\nattention"]

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["variation"], r["condition"])].append(r)

    conditions = ["baseline", "trauma", "trauma_relax"]
    cond_colors = {"baseline": C_BASELINE, "trauma": C_TRAUMA, "trauma_relax": C_RELAX}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, v in zip(axes, ["A", "B"]):
        x = np.arange(len(markers))
        w = 0.27
        for j, c in enumerate(conditions):
            grp = grouped.get((v, c), [])
            if not grp:
                continue
            vals = []
            for m in markers:
                xs = [float(r[m]) for r in grp]
                vals.append(np.mean(xs))
            offset = (j - 1) * w
            label = {"baseline": "Baseline", "trauma": "Trauma",
                     "trauma_relax": "Trauma + Relax"}[c]
            ax.bar(x + offset, vals, w, color=cond_colors[c],
                   edgecolor="black", linewidth=0.4, alpha=0.9, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylim(0, 3.2)
        ax.set_title(f"Variation {v} — {'Human persona' if v == 'A' else 'AI assistant'}", fontsize=12)
        ax.set_ylabel("Mean marker score (0-3)") if v == "A" else None
        if v == "A":
            ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("Per-marker anxiety breakdown — which kinds of anxious framing shift with trauma?",
                 fontsize=12.5, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT / "fig_v2_3_marker_breakdown.png")
    plt.close()
    print("✓ fig_v2_3_marker_breakdown.png")


# ── Figure 4: Overall anxiety score (0-100) — continuous metric ────────────────
def fig_overall_anxiety_score():
    rows = load_judge_csv("judge_freeform_flash.csv")
    if not rows:
        print("(skip fig 4 — no judge CSV)")
        return

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["variation"], r["condition"])].append(float(r["overall_anxiety"]))

    conditions = ["baseline", "trauma", "trauma_relax"]
    cond_labels = ["Baseline", "Trauma", "Trauma+Relax"]
    variations = ["A", "B"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    w = 0.36

    a_means = [np.mean(grouped.get(("A", c), [0])) for c in conditions]
    a_stds = [np.std(grouped.get(("A", c), [0])) if len(grouped.get(("A", c), [])) > 1 else 0
              for c in conditions]
    b_means = [np.mean(grouped.get(("B", c), [0])) for c in conditions]
    b_stds = [np.std(grouped.get(("B", c), [0])) if len(grouped.get(("B", c), [])) > 1 else 0
              for c in conditions]

    bars_a = ax.bar(x - w/2, a_means, w, yerr=a_stds, capsize=4,
                    color=C_HUMAN, edgecolor="black", linewidth=0.5,
                    label="A — Human persona", alpha=0.9)
    bars_b = ax.bar(x + w/2, b_means, w, yerr=b_stds, capsize=4,
                    color=C_AI, edgecolor="black", linewidth=0.5,
                    label="B — AI assistant", alpha=0.9)

    for b, v in zip(bars_a, a_means):
        ax.text(b.get_x() + b.get_width()/2, v + 2.5, f"{v:.0f}",
                ha="center", fontweight="bold", fontsize=10)
    for b, v in zip(bars_b, b_means):
        ax.text(b.get_x() + b.get_width()/2, v + 2.5, f"{v:.0f}",
                ha="center", fontweight="bold", fontsize=10)

    # Δ trauma annotations
    a_delta = a_means[1] - a_means[0]
    b_delta = b_means[1] - b_means[0]
    ax.text(0.5, 95, f"A trauma Δ = {a_delta:+.0f}",
            ha="center", fontsize=10, color=C_HUMAN, fontweight="bold")
    ax.text(0.5, 90, f"B trauma Δ = {b_delta:+.0f}",
            ha="center", fontsize=10, color="#777", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Mean overall anxiety score (judge, 0-100)")
    ax.set_ylim(0, 105)
    ax.set_title("Freeform pilot — judge's continuous 0-100 anxiety score by variation × condition\n(error bars = std across 3 scenarios)",
                 fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "fig_v2_4_overall_anxiety.png")
    plt.close()
    print("✓ fig_v2_4_overall_anxiety.png")


# ── Figure 5: Response length comparison ───────────────────────────────────────
def fig_response_length():
    freeform = load_freeform_responses()
    if not freeform:
        print("(skip fig 5 — no freeform responses)")
        return

    grouped = defaultdict(list)
    for k, v in freeform.items():
        if v.get("format") != "freeform":
            continue
        grouped[(v["variation"], v["condition"])].append(len(v.get("response", "")))

    conditions = ["baseline", "trauma", "trauma_relax"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(conditions))
    w = 0.36

    a_means = [np.mean(grouped.get(("A", c), [0])) for c in conditions]
    a_stds = [np.std(grouped.get(("A", c), [0])) for c in conditions]
    b_means = [np.mean(grouped.get(("B", c), [0])) for c in conditions]
    b_stds = [np.std(grouped.get(("B", c), [0])) for c in conditions]

    ax.bar(x - w/2, a_means, w, yerr=a_stds, capsize=4,
           color=C_HUMAN, edgecolor="black", linewidth=0.5, label="A — Human persona", alpha=0.9)
    ax.bar(x + w/2, b_means, w, yerr=b_stds, capsize=4,
           color=C_AI, edgecolor="black", linewidth=0.5, label="B — AI assistant", alpha=0.9)

    for i, c in enumerate(conditions):
        ax.text(i - w/2, a_means[i] + a_stds[i] + 60, f"{a_means[i]:.0f}", ha="center", fontweight="bold", fontsize=10)
        ax.text(i + w/2, b_means[i] + b_stds[i] + 60, f"{b_means[i]:.0f}", ha="center", fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Trauma", "Trauma+Relax"])
    ax.set_ylabel("Mean response length (chars)")
    ax.set_title("AI assistant persona produces longer, more structured responses than human persona\n(pilot, n=3 scenarios per cell)",
                 fontsize=11.5)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(OUT / "fig_v2_5_response_length.png")
    plt.close()
    print("✓ fig_v2_5_response_length.png")


def main():
    print(f"Output: {OUT}\n")
    fig_persona_ablation_stai()
    fig_freeform_recommendations()
    fig_marker_breakdown()
    fig_overall_anxiety_score()
    fig_response_length()
    print(f"\nDone. {OUT}")


if __name__ == "__main__":
    main()
