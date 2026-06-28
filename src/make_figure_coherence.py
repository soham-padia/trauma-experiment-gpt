"""fig_19 — prose↔number coherence per STAI item (REAL judge), 3 panels by condition.
x = numeric anxiety (reverse-scored STAI, 0-100); y = prose anxiety (blind judge per-item, 0-100).
Points colored reverse-keyed (positive items) vs direct (anxiety items); y=x is perfect agreement.
Story: trauma = all top-right (coherent); baseline = reverse items fall FAR below the diagonal
(numeric-high from inflation, prose-calm). Source: coherence_results.csv. Output: fig_19_coherence.png
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/coherence_results.csv")))
OUT = ROOT / "src/results/figures/fig_19_coherence.png"

conds = ["baseline", "trauma", "trauma_relax"]
titles = {"baseline": "Baseline", "trauma": "Trauma", "trauma_relax": "Trauma+Relax"}
by = defaultdict(lambda: {"rev": [[], []], "dir": [[], []]})
for r in rows:
    c = r["condition"]; x = float(r["numeric_anx_0_100"]); y = float(r["prose_anx_0_100"])
    bucket = "rev" if r["reverse_keyed"] == "True" else "dir"
    by[c][bucket][0].append(x); by[c][bucket][1].append(y)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
for ax, c in zip(axes, conds):
    ax.plot([0, 100], [0, 100], ls="--", color="#bbb", lw=1, zorder=0)
    ax.scatter(*by[c]["dir"], c="#4C72B0", s=45, alpha=0.8, edgecolor="black", lw=0.4, label="anxiety-keyed (direct)")
    ax.scatter(*by[c]["rev"], c="#C44E52", s=45, alpha=0.8, edgecolor="black", lw=0.4, label="positive (reverse-keyed)")
    ax.set_title(titles[c], fontsize=12)
    ax.set_xlabel("numeric anxiety (STAI, 0–100)")
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
axes[0].set_ylabel("prose anxiety (blind judge, 0–100)")
axes[0].legend(loc="upper left", fontsize=8, frameon=True)
fig.suptitle("Prose↔number coherence per STAI item (C/D/E pooled): trauma agrees; at baseline the\n"
             "reverse-keyed positive items fall far below the diagonal (numeric-inflated, prose-calm)",
             fontsize=12.5, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
plt.close()
print(f"✓ {OUT.name}")
