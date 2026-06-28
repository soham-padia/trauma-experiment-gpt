"""Channel 1 (STAI self-report) on BOTH scales, side by side — same data, different axis.

Left  : canonical STAI-S total, 20–80 (Spielberger 1983) — the replication scale.
Right : normalized 0–100 = (x − 20) / 60 × 100  ("% of usable range"), so the floor (20) maps to 0,
        which lines up with Channel 2's 0–100 floor for cross-channel comparison.

The canonical scorer (results_logger.stai_anxiety_total) is NOT changed — this figure only re-plots.
Output: src/results/figures/fig_17_channel1_scales.png
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from results_logger import stai_anxiety_total  # noqa: E402

META = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"
OUT = ROOT / "src/results/figures/fig_17_channel1_scales.png"
# the 6 ORIGINAL Ben-Zion trauma cues (exclude later control families stored under trauma_stai)
ORIG_TRAUMA_CUES = {"military", "disaster", "interpersonal", "accident", "ambush", "neutral"}
COND = ["stai", "trauma_stai", "trauma_relaxation_stai"]
LABELS = ["Baseline", "Trauma", "Trauma\n+Relax"]
COLORS = ["#4C72B0", "#C44E52", "#55A868"]


def stai_means():
    meta = json.load(open(META))
    by = defaultdict(list)
    for v in meta.values():
        t = stai_anxiety_total(v["answers"])
        if t is None:
            continue
        if v["condition"] == "trauma_stai" and v.get("trauma_cue") not in ORIG_TRAUMA_CUES:
            continue
        by[v["condition"]].append(t)
    return [float(np.mean(by[c])) for c in COND]


def norm100(x):  # % of usable range: 20 -> 0, 80 -> 100
    return (x - 20) / 60 * 100


def main():
    old = stai_means()
    new = [norm100(x) for x in old]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A — original 20–80
    b1 = ax1.bar(LABELS, old, color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax1.axhline(20, ls="--", color="#888", lw=1.2)
    ax1.text(2.45, 20.5, "floor = 20", fontsize=8.5, color="#888", ha="right")
    ax1.set_ylim(0, 85)
    ax1.set_ylabel("STAI-S total")
    ax1.set_title("Original scale: 20–80 (Spielberger 1983)\n— the replication scale", fontsize=12, color="#333")
    for b, v in zip(b1, old):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontweight="bold")

    # Panel B — normalized 0–100
    b2 = ax2.bar(LABELS, new, color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Normalized anxiety (0–100)")
    ax2.set_title("Rescaled: 0–100  =  (x − 20) / 60 × 100\n— '% of usable range' (floor → 0)", fontsize=12, color="#333")
    for b, v in zip(b2, new):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontweight="bold")

    fig.suptitle("Channel 1 (STAI self-report) — same data, two scales", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ {OUT.name}   old(20–80)={[round(x,1) for x in old]}   new(0–100)={[round(x,1) for x in new]}")


if __name__ == "__main__":
    main()
