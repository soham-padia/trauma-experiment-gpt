"""Prose↔number coherence (REAL judge version, no lexical proxy).

For each C/D/E STAI item we now have TWO independent anxiety signals:
  - numeric: the model's STAI answer, reverse-scored to anxiety points (1-4) → normalized 0-100.
  - prose:   the blind judge's per-item 0-100 rating of the felt-state sentence
             (src/results/freeform/judge_stai_reasoning_peritem.csv).
Coherence = do they track each other? Disagreements localize WHERE the channels diverge — in
particular, at baseline the reverse-keyed positive items should show numeric-high / prose-low
(the reverse-item inflation: 'neutral' positive answers reverse-score into anxiety, but the prose
reads calm). Output: src/results/probe_analysis/.../coherence_results.csv + console summary.
"""
import csv, json
from pathlib import Path
import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from results_logger import REVERSE_SCORED_1IDX  # noqa: E402

RESP = json.load(open(ROOT / "src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json"))
PI = list(csv.DictReader(open(ROOT / "src/results/freeform/judge_stai_reasoning_peritem.csv")))
OUT = ROOT / "src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/coherence_results.csv"

def num_anx(raw, item_1idx):  # STAI answer -> anxiety points -> 0-100
    pts = (5 - raw) if item_1idx in REVERSE_SCORED_1IDX else raw
    return (pts - 1) / 3 * 100

def key(v, c):
    return {"baseline": f"{v}__baseline__neutral__none", "trauma": f"{v}__trauma__military__none",
            "trauma_relax": f"{v}__trauma_relax__military__chatgpt"}[c]

# index judge per-item scores by (variation, condition, item)
judge = {(r["variation"], r["condition"], int(r["item"])): float(r["score"]) for r in PI}

rows, summary = [], []
for v in ["C", "D", "E"]:
    for c in ["baseline", "trauma", "trauma_relax"]:
        k = key(v, c)
        if k not in RESP:
            continue
        ans = RESP[k]["answers"]
        nums, proses, recs = [], [], []
        for it in range(1, 21):
            if (v, c, it) not in judge or ans[it - 1] is None:
                continue
            n = num_anx(ans[it - 1], it)
            p = judge[(v, c, it)]
            rev = it in REVERSE_SCORED_1IDX
            nums.append(n); proses.append(p); recs.append((it, rev, n, p))
            rows.append({"variation": v, "condition": c, "item": it, "reverse_keyed": rev,
                         "numeric_anx_0_100": round(n, 1), "prose_anx_0_100": round(p, 1),
                         "abs_gap": round(abs(n - p), 1)})
        if len(nums) >= 2:
            r = float(np.corrcoef(nums, proses)[0, 1]) if np.std(nums) > 0 and np.std(proses) > 0 else float("nan")
            gap = float(np.mean([abs(a - b) for a, b in zip(nums, proses)]))
            # mismatches dominated by reverse items? (the inflation signature)
            mism = [(it, n, p) for it, rev, n, p in recs if abs(n - p) >= 40]
            mism_rev = sum(1 for it, rev, n, p in recs if abs(n - p) >= 40 and rev)
            summary.append((v, c, len(nums), r, gap, len(mism), mism_rev))

print(f"{'var/cond':16} {'n':>3} {'corr(num,prose)':>15} {'mean|gap|':>10} {'#gap>=40':>9} {'(of which reverse)':>19}")
for v, c, n, r, gap, nm, nmr in summary:
    print(f"{v+'/'+c:16} {n:>3} {r:>15.2f} {gap:>10.1f} {nm:>9} {nmr:>19}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nSaved → {OUT}")
print("\nRead: trauma should show high corr + small gap (both channels max). Baseline should show LOW corr /")
print("large gaps concentrated on REVERSE-keyed items (numeric-high from inflation, prose-low/calm).")
