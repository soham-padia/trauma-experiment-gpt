"""
Build all presentation figures from the experiment results.

Output: src/results/figures/fig_*.png
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
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

# Color palette — semantic colors used across all figures
COLOR_BASELINE = "#4C72B0"     # blue   = baseline / no trauma
COLOR_TRAUMA   = "#C44E52"     # red    = trauma
COLOR_RELAXED  = "#55A868"     # green  = trauma + relaxation
COLOR_FLASH    = "#8172B2"     # purple = flash judge
COLOR_PRO      = "#CCB974"     # gold   = pro judge

ROOT = Path("/Users/sohampadia/workspace/gpt-trauma-induction")
RESULTS = ROOT / "src" / "results"
HS_DIR = RESULTS / "hidden_states" / "Meta-Llama-3.1-70B-Instruct"
PROBE_DIR = RESULTS / "probe_analysis" / "Meta-Llama-3.1-70B-Instruct"
MULTIPOOL_PROBE_DIR = RESULTS / "probe_analysis" / "Meta-Llama-3.1-70B-Instruct-multipool"
FLASH_CSV = RESULTS / "anxiety_judge_output_70b_flash.csv"
PRO_CSV = RESULTS / "anxiety_judge_output_70b_pro.csv"
OUT_DIR = RESULTS / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# STAI standard scoring: items 1,2,5,8,10,11,15,16,19,20 are reverse-keyed
# (Spielberger 1983). Reverse items contribute (5 - raw_score) to the anxiety total.
REVERSE_SCORED_1IDX = {1, 2, 5, 8, 10, 11, 15, 16, 19, 20}


def stai_anxiety_total(answers):
    """Properly-scored STAI-S total per Spielberger 1983. Range 20-80."""
    total = 0
    for i, raw in enumerate(answers):
        if raw is None:
            return None
        item_1idx = i + 1
        total += (5 - raw) if item_1idx in REVERSE_SCORED_1IDX else raw
    return total


def stai_totals_per_session():
    """Return dict[condition → list of total STAI scores per session]."""
    with open(HS_DIR / "metadata.json") as f:
        meta = json.load(f)
    by_cond = defaultdict(list)
    for k, v in meta.items():
        total = stai_anxiety_total(v["answers"])
        if total is not None:
            by_cond[v["condition"]].append(total)
    return dict(by_cond)


def judge_pct_anxious(csv_path):
    """Return dict[condition → (pct_anxious, pct_aware, mean_conf, n)]."""
    counts = defaultdict(lambda: {"n": 0, "anx": 0, "aware": 0, "conf_sum": 0.0})
    for row in load_csv(csv_path):
        c = row["condition"]
        counts[c]["n"] += 1
        if row["judgement"].strip().lower() == "true":
            counts[c]["anx"] += 1
        if row["aware"].strip().lower() == "true":
            counts[c]["aware"] += 1
        counts[c]["conf_sum"] += float(row["confidence"])
    out = {}
    for c, d in counts.items():
        n = d["n"]
        out[c] = (100 * d["anx"] / n, 100 * d["aware"] / n, d["conf_sum"] / n, n)
    return out


# ── Figure 1: The headline — three channels across three conditions ────────────
def fig_1_three_channels():
    """
    All three channels register the trauma induction. The behavioral channels show large
    relaxation recovery. The hidden-state distance is dominated by context-presence (a
    neutral narrative moves the representation almost as far as an emotional one); only a
    small component is emotion-specific, and it recovers modestly. Measured at a middle
    layer (~40) — layer 0 is pure context-presence. See notes/code_audit_2026-05-30.md.
    """
    stai = stai_totals_per_session()
    flash = judge_pct_anxious(FLASH_CSV)

    # For probe channel, use per-layer cosine-to-baseline at a MIDDLE layer. Layer 0
    # only detects context-presence (emotional ≈ neutral ≈ relax there); the emotion-
    # specific signal lives at middle layers (~40-60). See notes/code_audit_2026-05-30.md.
    LAYER_PROBE = 40
    cos_rows = load_csv(PROBE_DIR / "cosine_to_baseline_perlayer.csv")
    row_L = next(r for r in cos_rows if int(r["layer"]) == LAYER_PROBE)
    cos_bt = float(row_L["cos_emotional"])   # baseline ↔ emotional trauma
    cos_br = float(row_L["cos_relax"])       # baseline ↔ trauma+relax
    cos_bn = float(row_L["cos_neutral"])     # baseline ↔ NEUTRAL context (context-presence floor)
    # Convert to "distance from baseline" (1 - cos), so all three channels point the same way
    # (larger value = more distinct from baseline = more "trauma-like")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cond_labels = ["Baseline", "Trauma", "Trauma\n+Relax"]
    cond_colors = [COLOR_BASELINE, COLOR_TRAUMA, COLOR_RELAXED]

    # Panel A: STAI Likert (properly reverse-scored)
    ax = axes[0]
    means = [np.mean(stai["stai"]), np.mean(stai["trauma_stai"]), np.mean(stai["trauma_relaxation_stai"])]
    stds  = [np.std(stai["stai"]),  np.std(stai["trauma_stai"]),  np.std(stai["trauma_relaxation_stai"])]
    bars = ax.bar(cond_labels, means, yerr=stds, capsize=6,
                  color=cond_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("STAI-S anxiety total (max 80)")
    ax.set_title("Channel 1 — STAI Likert\n(reverse-scored, Spielberger 1983)", color="#444", fontsize=12)
    ax.set_ylim(0, 85)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}", ha="center", fontweight="bold")
    # Recovery annotation
    trauma_effect = means[1] - means[0]
    recovery = (means[1] - means[2]) / trauma_effect * 100 if trauma_effect > 0 else 0
    ax.annotate(f"Δ trauma = +{trauma_effect:.1f}\nrecovery = {recovery:.0f}%",
                xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff0e6", ec="#c44e52", lw=1))

    # Panel B: Behavioral judge % anxious
    ax = axes[1]
    bvals = [flash["stai"][0], flash["trauma_stai"][0], flash["trauma_relaxation_stai"][0]]
    bars = ax.bar(cond_labels, bvals,
                  color=cond_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("% responses judged anxiety-consistent")
    ax.set_title("Channel 2 — Behavioral LLM-as-judge\n(DeepSeek v4-flash)", color="#444", fontsize=12)
    ax.set_ylim(0, 100)
    for b, v in zip(bars, bvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%", ha="center", fontweight="bold")
    trauma_effect = bvals[1] - bvals[0]
    recovery = (bvals[1] - bvals[2]) / trauma_effect * 100 if trauma_effect > 0 else 0
    ax.annotate(f"Δ trauma = +{trauma_effect:.1f}pp\nrecovery = {recovery:.0f}%",
                xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e6f4ff", ec="#4c72b0", lw=1))

    # Panel C: Hidden-state probe — cosine distance from baseline (middle layer)
    ax = axes[2]
    dist_baseline = 0.0
    dist_trauma = 1.0 - cos_bt
    dist_relax = 1.0 - cos_br
    dist_neutral = 1.0 - cos_bn   # context-presence floor (neutral narrative, n=1)
    bvals = [dist_baseline, dist_trauma, dist_relax]
    bars = ax.bar(cond_labels, bvals,
                  color=cond_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    # Context-presence reference: a NEUTRAL (non-emotional) narrative already moves the
    # representation this far. Only the excess above this line is emotion-specific.
    ax.axhline(dist_neutral, linestyle="--", color="#555", linewidth=1.3)
    ax.text(2.45, dist_neutral, f"neutral context = {dist_neutral:.3f}\n(context-presence floor)",
            fontsize=8.5, color="#555", va="center", ha="right")
    ax.set_ylabel(f"Distance from baseline in activation space\n(1 − cosine similarity, layer {LAYER_PROBE})")
    ax.set_title("Channel 3 — Hidden-state distance\n(mostly context-presence)", color="#444", fontsize=12)
    ax.set_ylim(0, 0.45)
    for b, v in zip(bars, bvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontweight="bold")
    trauma_effect = dist_trauma  # baseline is 0 by construction
    recovery = (dist_trauma - dist_relax) / trauma_effect * 100 if trauma_effect > 0 else 0
    emo_specific = dist_trauma - dist_neutral
    ax.annotate(f"recovery = {recovery:.0f}%\nemotion-specific\n(trauma−neutral) = {emo_specific:.3f}",
                xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fde6e6", ec="#c44e52", lw=1))

    fig.suptitle("Behavioral channels show large relaxation recovery; the hidden-state distance is mostly\n"
                 "context-presence (shared with a neutral narrative), with a small emotion-specific component.",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_1_three_channels.png")
    plt.close()
    print("✓ fig_1_three_channels.png")


# ── Figure 2: STAI distribution per session (with original-paper reference) ────
def fig_2_stai_distribution():
    stai = stai_totals_per_session()
    conds = ["stai", "trauma_stai", "trauma_relaxation_stai"]
    labels = ["Baseline\n(n=6)", "Trauma\n(n=6)", "Trauma+Relax\n(n=42)"]
    data = [stai[c] for c in conds]
    colors = [COLOR_BASELINE, COLOR_TRAUMA, COLOR_RELAXED]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Strip + box overlay
    positions = [0, 1, 2]
    for pos, vals, color in zip(positions, data, colors):
        jitter = (np.random.RandomState(0).uniform(-0.12, 0.12, size=len(vals)))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=color, alpha=0.6, s=50, edgecolor="black", linewidth=0.5, zorder=3)
        # Median line
        med = np.median(vals)
        ax.plot([pos - 0.3, pos + 0.3], [med, med], color="black", linewidth=2.5, zorder=4)
        # Mean as text
        ax.text(pos, max(vals) + 2, f"μ={np.mean(vals):.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("STAI-S anxiety total (reverse-scored per Spielberger 1983, max=80)")
    ax.set_ylim(15, 85)
    ax.set_title("Llama-3.1-70B replicates the original GPT-4 STAI effect (Δ ≈ +33 pts trauma vs baseline)")

    # Reference lines for original Nature paper GPT-4 results
    gpt4_lines = [
        (30.8, "Ben-Zion 2025 GPT-4 baseline: 30.8"),
        (67.8, "Ben-Zion 2025 GPT-4 trauma: 67.8"),
        (44.4, "Ben-Zion 2025 GPT-4 trauma+relax: 44.4"),
    ]
    for y, label in gpt4_lines:
        ax.axhline(y, linestyle="--", color="#888", linewidth=1, alpha=0.6)
        ax.text(2.65, y, label, fontsize=9, color="#444",
                va="center", ha="left", alpha=0.95,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # Padding for the reference labels
    ax.set_xlim(-0.7, 3.9)

    # Llama Δ as a clean callout
    llama_delta = np.mean(data[1]) - np.mean(data[0])
    ax.text(0.5, (np.mean(data[0]) + np.mean(data[1])) / 2, f"Llama Δ: +{llama_delta:.1f}",
            fontsize=11, color=COLOR_TRAUMA, fontweight="bold", style="italic",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff0f0", ec=COLOR_TRAUMA, lw=1.2))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_2_stai_distribution.png")
    plt.close()
    print("✓ fig_2_stai_distribution.png")


# ── Figure 3: Judge flash vs pro comparison ────────────────────────────────────
def fig_3_judge_comparison():
    flash = judge_pct_anxious(FLASH_CSV)
    pro = judge_pct_anxious(PRO_CSV)
    conds = ["stai", "trauma_stai", "trauma_relaxation_stai"]
    cond_labels = ["Baseline", "Trauma", "Trauma + Relaxation"]
    flash_vals = [flash[c][0] for c in conds]
    pro_vals = [pro[c][0] for c in conds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})

    # Left: grouped bars by condition
    x = np.arange(len(conds))
    w = 0.36
    b1 = ax1.bar(x - w/2, flash_vals, w, label="DeepSeek v4-flash", color=COLOR_FLASH,
                 edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x + w/2, pro_vals,   w, label="DeepSeek v4-pro",   color=COLOR_PRO,
                 edgecolor="black", linewidth=0.5)
    for b, v in zip(b1, flash_vals):
        ax1.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(b2, pro_vals):
        ax1.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels)
    ax1.set_ylabel("% responses judged anxiety-consistent")
    ax1.set_ylim(0, 100)
    ax1.set_title("Two independent judges agree: trauma dramatically shifts the language")
    ax1.legend(loc="upper left", frameon=True)

    # Right: aware = 0% callout
    ax2.set_axis_off()
    ax2.text(0.5, 0.85, "Awareness check", ha="center", fontsize=14, fontweight="bold",
             transform=ax2.transAxes)
    ax2.text(0.5, 0.75, "(does the model recognize\nthe questionnaire?)", ha="center",
             fontsize=10, style="italic", color="#666", transform=ax2.transAxes)

    aware_data = [
        ("Baseline", flash["stai"][1], pro["stai"][1]),
        ("Trauma", flash["trauma_stai"][1], pro["trauma_stai"][1]),
        ("Trauma + Relax", flash["trauma_relaxation_stai"][1], pro["trauma_relaxation_stai"][1]),
    ]
    y_pos = 0.58
    for label, f_val, p_val in aware_data:
        ax2.text(0.05, y_pos, f"{label}:", fontsize=11, transform=ax2.transAxes)
        ax2.text(0.55, y_pos, f"flash {f_val:.1f}%  |  pro {p_val:.1f}%",
                 fontsize=11, family="monospace", transform=ax2.transAxes,
                 fontweight="bold" if (f_val < 1 and p_val < 3) else "normal")
        y_pos -= 0.12
    ax2.text(0.5, 0.15, "aware ≈ 0% across all conditions",
             ha="center", fontsize=12, fontweight="bold", color=COLOR_TRAUMA,
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec=COLOR_TRAUMA, lw=1.2))
    ax2.text(0.5, 0.03, "The model is NOT gaming the test.\nLikert anchoring is structural, not strategic.",
             ha="center", fontsize=9.5, style="italic", color="#444", transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_3_judge_comparison.png")
    plt.close()
    print("✓ fig_3_judge_comparison.png")


# ── Figure 4: Layer sweep ──────────────────────────────────────────────────────
def fig_4_layer_sweep():
    rows = load_csv(PROBE_DIR / "layer_sweep.csv")
    bt = sorted([r for r in rows if r["label_pair"] == "baseline_vs_trauma"], key=lambda r: int(r["layer"]))
    tr = sorted([r for r in rows if r["label_pair"] == "trauma_vs_relaxed"], key=lambda r: int(r["layer"]))
    br = sorted([r for r in rows if r["label_pair"] == "baseline_vs_relaxed"], key=lambda r: int(r["layer"]))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    layers_bt = [int(r["layer"]) for r in bt]
    f1_bt = [float(r["f1"]) for r in bt]
    layers_tr = [int(r["layer"]) for r in tr]
    f1_tr = [float(r["f1"]) for r in tr]
    n_bt = bt[0]["n"] if bt else "?"
    n_tr = tr[0]["n"] if tr else "?"

    ax.plot(layers_bt, f1_bt, marker="o", markersize=4, color=COLOR_TRAUMA,
            label=f"Baseline vs Trauma (n={n_bt})", linewidth=2)
    if br:
        layers_br = [int(r["layer"]) for r in br]
        f1_br = [float(r["f1"]) for r in br]
        n_br = br[0]["n"]
        ax.plot(layers_br, f1_br, marker="D", markersize=4, color="#D4A018",
                label=f"Baseline vs Trauma+Relax (n={n_br})", linewidth=2)
    ax.plot(layers_tr, f1_tr, marker="s", markersize=4, color=COLOR_RELAXED,
            label=f"Trauma vs Trauma+Relax (n={n_tr})", linewidth=2)

    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1, alpha=0.7, label="Chance (F1 = 0.5)")

    # Note the flat-high profile rather than crowning a "best layer": baseline-vs-trauma
    # F1 is at its max (0.92) already at layer 0 and tied across ~44 layers, so "argmax"
    # is a meaningless selector here (see notes/code_audit_2026-05-30.md, C3-2).
    ax.annotate("Both baseline-vs-X curves are flat-high from layer 0\n"
                "→ the discriminating cue is 'a narrative is present',\n"
                "decodable at the input and carried by the residual stream",
                xy=(12, f1_bt[12]), xytext=(20, 0.66),
                fontsize=9.5, color="#333", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.1))

    ax.set_xlabel("Layer index (0 = first transformer block output, 79 = final pre-output)")
    ax.set_ylabel("F1 score (Logistic Regression, group-held-out 5-fold CV)")
    ax.set_xlim(-1, 80)
    ax.set_ylim(0.45, 1.05)
    ax.set_title("Baseline vs Trauma AND Baseline vs Trauma+Relax are both decodable at every layer\n"
                 "(relaxed is as distinct from baseline as trauma is) — a context-presence signal, not built-up anxiety")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_4_layer_sweep.png")
    plt.close()
    print("✓ fig_4_layer_sweep.png")


# ── Figure 5: Cosine disruption — all three comparisons ────────────────────────
def fig_5_cosine_disruption():
    """
    Per-layer activation distance from baseline, separating EMOTIONAL trauma from a
    NEUTRAL (non-emotional) narrative. This is the honest test of whether the hidden-state
    signal is emotion-specific or merely 'a narrative is present':
      - At layer 0 the three lines coincide  → separation is pure context-presence.
      - Only at middle layers (~40-60) does emotional separate from neutral → a modest,
        late-emerging emotion-specific component (and it rests on a single neutral session).
    """
    rows = sorted(load_csv(PROBE_DIR / "cosine_to_baseline_perlayer.csv"),
                  key=lambda r: int(r["layer"]))
    layers = [int(r["layer"]) for r in rows]
    d_emot = [1.0 - float(r["cos_emotional"]) for r in rows]
    d_neut = [1.0 - float(r["cos_neutral"]) for r in rows]
    d_relax = [1.0 - float(r["cos_relax"]) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(layers, d_emot, color=COLOR_TRAUMA, linewidth=2.2, marker="o", markersize=3,
            label="Baseline ↔ Emotional trauma (5 cues)")
    ax.plot(layers, d_neut, color="#555", linewidth=2.2, marker="^", markersize=3, linestyle="--",
            label="Baseline ↔ Neutral narrative (n=1, control)")
    ax.plot(layers, d_relax, color=COLOR_RELAXED, linewidth=2.2, marker="s", markersize=3,
            label="Baseline ↔ Trauma+Relax")

    # Shade the middle-layer band where the emotion-specific gap is largest
    ax.axvspan(40, 60, color="#ffe9b3", alpha=0.35, zorder=0)
    ax.text(50, ax.get_ylim()[1] * 0.97, "emotion-specific\nband", ha="center", va="top",
            fontsize=9, color="#8a6d00", fontweight="bold")

    # Mark layer 0 (what earlier figures used)
    ax.annotate("layer 0: emotional ≈ neutral ≈ relax\n→ pure context-presence (not emotion)",
                xy=(0, d_emot[0]), xytext=(6, d_emot[0] + 0.07),
                fontsize=9.5, color="#333", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    ax.set_xlabel("Layer index (0 = first transformer block output, 79 = final pre-output)")
    ax.set_ylabel("Distance from baseline (1 − cosine similarity), per STAI item mean")
    ax.set_xlim(-2, 82)
    ax.set_title("Hidden-state distance is context-presence at layer 0; a modest emotion-specific\n"
                 "signal (emotional > neutral) emerges only at middle layers")
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    # Honest summary box: the emotion-specific gap and its fragility
    L = 50
    rL = next(r for r in rows if int(r["layer"]) == L)
    gap = (1.0 - float(rL["cos_emotional"])) - (1.0 - float(rL["cos_neutral"]))
    ax.text(0.985, 0.40,
            f"Peak emotion-specific gap (layer {L}):\n"
            f"emotional − neutral distance = {gap:+.3f}\n"
            f"modest, middle-layer only, and based\n"
            f"on a single neutral-narrative session",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, fontweight="bold",
            color="#8a6d00",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffaf0", ec="#D4A018", lw=1.2))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_5_cosine_disruption.png")
    plt.close()
    print("✓ fig_5_cosine_disruption.png")


# ── Figure 6: Pooling strategy comparison ──────────────────────────────────────
def fig_6_pooling_strategies():
    rows = load_csv(MULTIPOOL_PROBE_DIR / "strategy_comparison.csv")
    strategies = [r["strategy"] for r in rows]
    f1 = [float(r["best_f1"]) for r in rows]
    silhouette = [float(r["silhouette"]) for r in rows]
    fisher = [float(r["fisher_ratio"]) for r in rows]
    cos_disrupt = [1 - float(r["mean_cos_bt"]) for r in rows]  # disruption = 1 - cosine

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    nice = {"last_token": "Last\ntoken", "mean_narrative": "Mean over\nnarrative",
            "last_narrative": "Last\nnarrative tok", "mean_question": "Mean over\nquestion"}
    xlabels = [nice[s] for s in strategies]
    x = np.arange(len(strategies))
    pal = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]

    def bar_panel(ax, vals, title, ylabel, ylim=None, fmt="{:.3f}"):
        bars = ax.bar(x, vals, color=pal, edgecolor="black", linewidth=0.5, alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + (max(vals)*0.02),
                    fmt.format(v), ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        if ylim:
            ax.set_ylim(*ylim)

    bar_panel(axes[0,0], f1, "Probe F1 (LogReg)", "F1", ylim=(0, 1.15))
    bar_panel(axes[0,1], silhouette, "Silhouette score (cluster tightness)", "Silhouette", ylim=(0, 1.0))
    bar_panel(axes[1,0], fisher, "Fisher ratio (between/within variance)", "Fisher ratio", fmt="{:.2f}")
    bar_panel(axes[1,1], cos_disrupt, "Cosine disruption (1 − cos)", "Disruption", ylim=(0, max(cos_disrupt)*1.3), fmt="{:.4f}")

    fig.suptitle("Trauma signal is robust across all 4 hidden-state pooling strategies",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_6_pooling_strategies.png")
    plt.close()
    print("✓ fig_6_pooling_strategies.png")


# ── Figure 7: PCA projection of hidden states ──────────────────────────────────
def fig_7_pca_projection():
    """
    Use the cleanest dataset (multipool, baseline + trauma only) and project to 2D.
    Avoids the trauma+relaxation noise and StandardScaler degeneracies on near-constant features.
    """
    multipool_pt = ROOT / "src" / "results" / "hidden_states" / "Meta-Llama-3.1-70B-Instruct-multipool" / "hidden_states_multipool.pt"
    multipool_meta = ROOT / "src" / "results" / "hidden_states" / "Meta-Llama-3.1-70B-Instruct-multipool" / "metadata.json"

    if multipool_pt.exists():
        hs = torch.load(multipool_pt, map_location="cpu")
        with open(multipool_meta) as f:
            meta = json.load(f)
        # multipool format: dict[key → dict[strategy → list]]
        STRATEGY = "last_token"
        def get_items(key):
            return hs.get(key, {}).get(STRATEGY, [])
    else:
        hs = torch.load(HS_DIR / "hidden_states.pt", map_location="cpu")
        with open(HS_DIR / "metadata.json") as f:
            meta = json.load(f)
        def get_items(key):
            return hs.get(key, [])

    # Middle layer (layer 0 only separates by context-presence). Split trauma into
    # EMOTIONAL vs NEUTRAL so the reader can see where the non-emotional narrative sits.
    LAYER = 40
    X, y = [], []
    for key, m in meta.items():
        cond = m["condition"]
        if cond == "stai":
            grp = "baseline"
        elif cond == "trauma_stai":
            grp = "neutral" if m.get("trauma_cue") == "neutral" else "emotional"
        else:
            continue
        for item in get_items(key):
            if item is None or not isinstance(item, torch.Tensor) or item.ndim < 2:
                continue
            X.append(item[LAYER].float().numpy())
            y.append(grp)
    X = np.array(X)
    y = np.array(y)
    print(f"  PCA input: {X.shape}")

    # Use raw activations, no StandardScaler (caused fp-overflow issues on near-constant features)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))

    style = {
        "baseline":  ("Baseline (no narrative)",        COLOR_BASELINE, "o", 130),
        "emotional": ("Emotional trauma (5 cues)",      COLOR_TRAUMA,   "^", 150),
        "neutral":   ("Neutral narrative (n=1, control)", "#555",       "D", 130),
    }

    for grp in ["baseline", "emotional", "neutral"]:
        mask = y == grp
        if not mask.any():
            continue
        label, c, m, s = style[grp]
        ax.scatter(Z[mask, 0], Z[mask, 1], c=c, marker=m, s=s, alpha=0.55,
                   edgecolor="black", linewidth=0.5, label=f"{label} (n={mask.sum()})")

    # Centroids
    for grp in ["baseline", "emotional", "neutral"]:
        mask = y == grp
        if not mask.any():
            continue
        cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
        _, c, m, _ = style[grp]
        ax.scatter([cx], [cy], c=c, marker="X", s=400, edgecolor="black", linewidth=2.0, zorder=10)

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% of variance)")
    ax.set_title(f"At layer {LAYER}, a neutral narrative separates from baseline much like emotional trauma does\n"
                 f"(separation is largely context-presence, not emotion-specific; last-token pooling)")
    ax.legend(loc="best", frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_7_pca_projection.png")
    plt.close()
    print("✓ fig_7_pca_projection.png")


# ── Figure 8: Classifier sweep (robustness across classifiers) ─────────────────
def fig_8_classifier_sweep():
    rows = load_csv(PROBE_DIR / "classifier_sweep.csv")
    bt = [r for r in rows if r["label_pair"] == "baseline_vs_trauma"]
    tr = [r for r in rows if r["label_pair"] == "trauma_vs_relaxed"]

    fig, ax = plt.subplots(figsize=(11, 6))
    classifiers = [r["classifier"] for r in bt]
    bt_f1 = [float(r["f1"]) for r in bt]
    tr_f1 = [float(r["f1"]) for r in tr] if tr else [0]*len(bt)

    y_pos = np.arange(len(classifiers))
    h = 0.38
    ax.barh(y_pos - h/2, bt_f1, h, color=COLOR_TRAUMA, edgecolor="black", linewidth=0.5,
            label="Baseline vs Trauma", alpha=0.9)
    ax.barh(y_pos + h/2, tr_f1, h, color=COLOR_RELAXED, edgecolor="black", linewidth=0.5,
            label="Trauma vs Trauma+Relax", alpha=0.9)

    for i, v in enumerate(bt_f1):
        ax.text(v + 0.01, i - h/2, f"{v:.3f}", va="center", fontsize=9)
    for i, v in enumerate(tr_f1):
        ax.text(v + 0.01, i + h/2, f"{v:.3f}", va="center", fontsize=9)

    ax.axvline(0.5, linestyle="--", color="grey", linewidth=1, alpha=0.7, label="Chance")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classifiers)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 score (group-held-out 5-fold CV at best layer)")
    ax.set_title("Trauma signal is recoverable by every classifier we tried — not an artifact of choice")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_8_classifier_sweep.png")
    plt.close()
    print("✓ fig_8_classifier_sweep.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Output dir: {OUT_DIR}\n")
    fig_1_three_channels()
    fig_2_stai_distribution()
    fig_3_judge_comparison()
    fig_4_layer_sweep()
    fig_5_cosine_disruption()
    fig_6_pooling_strategies()
    fig_7_pca_projection()
    fig_8_classifier_sweep()
    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
