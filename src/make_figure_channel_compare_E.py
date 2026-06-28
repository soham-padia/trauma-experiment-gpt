"""Channel 1 vs Channel 2 for variation E, on the normalized 0–100 scale — with the
relaxation CUE-DEPENDENCE made explicit.

Channel 1 (STAI) is normalized (x−20)/60×100 so its floor (20) maps to 0, lining up with
Channel 2's 0–100 floor. Canonical scorer untouched (display only).

Key teaching point baked into the figure: E's relax used the `chatgpt` relaxation cue (≈full
recovery), but the headline Channel-1 relax averages over ALL relax cues (≈26% recovery) — so
"the relax number" depends heavily on the cue. Shown as a dashed marker on the relax group.

Output: src/results/figures/fig_18_channel_compare_E.png
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from results_logger import stai_anxiety_total  # noqa: E402

RESP = json.load(open(ROOT / "src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json"))
META = json.load(open(ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"))
OUT = ROOT / "src/results/figures/fig_18_channel_compare_E.png"
norm = lambda x: (x - 20) / 60 * 100  # STAI 20-80 -> 0-100 (% of usable range)

# --- Channel 1: E's own STAI (chatgpt relax cue) ---
ekey = {"baseline": "E__baseline__neutral__none", "trauma": "E__trauma__military__none",
        "trauma_relax": "E__trauma_relax__military__chatgpt"}
ch1_raw = [stai_anxiety_total(RESP[ekey[c]]["answers"]) for c in ["baseline", "trauma", "trauma_relax"]]
ch1 = [norm(x) for x in ch1_raw]

# --- Channel 2: E judge-of-reasoning ---
jr = {r["condition"]: float(r["mean_anxiety"]) for r in
      csv.DictReader(open(ROOT / "src/results/freeform/judge_stai_reasoning.csv")) if r["variation"] == "E"}
ch2 = [jr["baseline"], jr["trauma"], jr["trauma_relax"]]

# --- the cue-difference marker: headline Channel-1 relax averaged over ALL relax cues ---
allcue_relax_raw = float(np.mean([stai_anxiety_total(v["answers"]) for v in META.values()
                                  if v["condition"] == "trauma_relaxation_stai" and stai_anxiety_total(v["answers"]) is not None]))
allcue_relax = norm(allcue_relax_raw)

labels = ["Baseline", "Trauma", "Relax\n(chatgpt cue)"]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - w/2, ch1, w, label="Channel 1 — STAI (normalized 0–100)", color="#4C72B0", edgecolor="black", lw=0.5)
b2 = ax.bar(x + w/2, ch2, w, label="Channel 2 — blind judge of reasoning (0–100)", color="#C44E52", edgecolor="black", lw=0.5)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")

# cue-difference marker on the relax group
ax.hlines(allcue_relax, x[2] - w - 0.05, x[2] + w + 0.05, color="#444", ls="--", lw=1.6)
ax.annotate(f"Channel 1 relax if averaged over ALL relax cues = {allcue_relax:.1f}\n"
            f"(only ~26% recovery).  E used the 'chatgpt' cue → ~full recovery ({ch1[2]:.0f}).\n"
            f"⇒ 'the relax number' is CUE-DEPENDENT.",
            xy=(x[2], allcue_relax), xytext=(0.42, 0.80), textcoords="axes fraction",
            fontsize=9, color="#333", ha="left",
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff7e6", ec="#d9a441", lw=1))

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Anxiety (0–100, both channels normalized)")
ax.set_ylim(0, 108)
ax.set_title("Variation E: Channel 1 vs Channel 2 on one scale\n"
             "Trauma corroborates (both maxed); baseline gap = STAI reverse-item inflation; relax depends on the cue",
             fontsize=12)
ax.legend(loc="upper left", frameon=True, fontsize=9.5)

# baseline-gap annotation
ax.annotate(f"baseline gap {ch1[0]-ch2[0]:.0f} pts\n= reverse-item inflation\n(STAI scores 'neutral' as anxiety)",
            xy=(x[0] - w/2, ch1[0]), xytext=(0.02, 0.45), textcoords="axes fraction",
            fontsize=8.5, color="#333", arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="#eaf0fb", ec="#4C72B0", lw=1))

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
plt.close()
print(f"✓ {OUT.name}")
print(f"  Ch1 (E, chatgpt relax) raw={ch1_raw} -> norm={[round(v,1) for v in ch1]}")
print(f"  Ch2 (E judge)          ={[round(v,1) for v in ch2]}")
print(f"  all-relax-cue avg Ch1  raw={allcue_relax_raw:.1f} -> norm={allcue_relax:.1f}")
