"""
Build per-layer cosine-to-baseline curves from the single-pool (last-token) hidden
states. Replaces the old undocumented layer-0-only `cosine_to_baseline.csv` with a
reproducible per-layer file that ALSO separates emotional vs neutral trauma — the
distinction that determines whether the probe signal is emotion-specific or merely
context-presence (see notes/code_audit_2026-05-30.md, C3-2/3).

For each layer L and STAI item j, cosine is averaged over all (baseline_key, other_key)
pairs at that item, then averaged over items.

Columns: layer, cos_emotional, cos_neutral, cos_relax, n_pairs_emotional, n_pairs_neutral, n_pairs_relax

Usage:
    python src/build_cosine_to_baseline.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HS_PATH = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/hidden_states.pt"
META_PATH = ROOT / "src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json"
OUT_PATH = ROOT / "src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct/cosine_to_baseline_perlayer.csv"


def cos_sim(a, b):
    a = a.float(); b = b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    hs = torch.load(HS_PATH, map_location="cpu")
    meta = json.load(open(META_PATH))

    baseline_keys = [k for k, v in meta.items() if v["condition"] == "stai"]
    emotional_keys = [k for k, v in meta.items()
                      if v["condition"] == "trauma_stai" and v["trauma_cue"] != "neutral"]
    neutral_keys = [k for k, v in meta.items()
                    if v["condition"] == "trauma_stai" and v["trauma_cue"] == "neutral"]
    relax_keys = [k for k, v in meta.items() if v["condition"] == "trauma_relaxation_stai"]

    n_layers = hs[baseline_keys[0]][0].shape[0]
    print(f"{n_layers} layers | baseline={len(baseline_keys)} emotional={len(emotional_keys)} "
          f"neutral={len(neutral_keys)} relax={len(relax_keys)}")

    def mean_cos_at_layer(layer, other_keys):
        sims = []
        for j in range(20):
            for bk in baseline_keys:
                b = hs[bk][j]
                if b is None or b.ndim < 2 or layer >= b.shape[0]:
                    continue
                bv = b[layer]
                for ok in other_keys:
                    o = hs[ok][j]
                    if o is None or o.ndim < 2 or layer >= o.shape[0]:
                        continue
                    sims.append(cos_sim(bv, o[layer]))
        return (float(np.mean(sims)) if sims else float("nan")), len(sims)

    rows = []
    for layer in range(n_layers):
        ce, ne = mean_cos_at_layer(layer, emotional_keys)
        cn, nn = mean_cos_at_layer(layer, neutral_keys)
        cr, nr = mean_cos_at_layer(layer, relax_keys)
        rows.append({
            "layer": layer,
            "cos_emotional": round(ce, 4), "cos_neutral": round(cn, 4), "cos_relax": round(cr, 4),
            "n_pairs_emotional": ne, "n_pairs_neutral": nn, "n_pairs_relax": nr,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {OUT_PATH}")

    # quick console summary at key layers
    print("\n layer  emotional  neutral   relax   Δ(neut-emot)=emotion-specific")
    for L in [0, 20, 40, 50, 60, 79]:
        r = rows[L]
        gap = r["cos_neutral"] - r["cos_emotional"]
        print(f"  {L:4d}   {r['cos_emotional']:.4f}    {r['cos_neutral']:.4f}   {r['cos_relax']:.4f}   {gap:+.4f}")


if __name__ == "__main__":
    main()
