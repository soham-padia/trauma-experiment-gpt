"""
Anxiety Probe Analysis — Multi-Pool Strategy Comparison

Loads hidden_states_multipool.pt (format: dict[key → dict[strategy → list[20 tensors]]])
and compares 4 hidden-state extraction strategies on probe performance.

Strategies:
  last_token      — hidden state at last input token before generation
  mean_narrative  — mean over trauma narrative / system-prompt tokens
  last_narrative  — last narrative token
  mean_question   — mean over STAI question tokens

For each strategy runs: layer sweep, best-layer classifier sweep, cosine disruption, geometry.
Outputs a summary comparison table plus per-strategy CSVs.

Usage:
    python src/anxiety_probe_multipool.py \
        --input-pt src/results/hidden_states/Meta-Llama-3.1-70B-Instruct-multipool/hidden_states_multipool.pt \
        --input-metadata src/results/hidden_states/Meta-Llama-3.1-70B-Instruct-multipool/metadata.json \
        --output-dir src/results/probe_analysis/Meta-Llama-3.1-70B-Instruct-multipool
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [
    str(REPO_ROOT / "pre-sycophancy-study" / "analysis"),
    str(REPO_ROOT / "pre-sycophancy-study"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preflip_geometry import fisher_ratio, lda_accuracy, silhouette  # noqa: E402
from cosine_disruption import cos_sim  # noqa: E402

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="lbfgs"),
    "Linear SVM":          lambda: LinearSVC(max_iter=2000, C=0.1, class_weight="balanced"),
    "LDA":                 lambda: LinearDiscriminantAnalysis(),
    "RBF SVM":             lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced"),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Extra Trees":         lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "KNN (k=5)":           lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Naive Bayes":         lambda: GaussianNB(),
    "MLP (128+64)":        lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True),
}

STRATEGIES = ["last_token", "mean_narrative", "last_narrative", "mean_question"]
CONDITION_LABEL = {"stai": 0, "trauma_stai": 1, "trauma_relaxation_stai": 2}


def run_cv(X, y, qids, clf_factory, n_splits=5):
    uq = np.unique(qids)
    ql = np.array([y[qids == qi].max() for qi in uq])
    if len(np.unique(ql)) < 2:
        return None, None
    actual_splits = min(n_splits, min(np.sum(ql == 0), np.sum(ql == 1)))
    if actual_splits < 2:
        return None, None
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    all_p, all_t = [], []
    for tr_idx, te_idx in skf.split(uq, ql):
        tr_qs = set(uq[tr_idx])
        te_qs = set(uq[te_idx])
        tr_mask = np.array([qi in tr_qs for qi in qids])
        te_mask = np.array([qi in te_qs for qi in qids])
        if len(np.unique(y[tr_mask])) < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_mask])
        Xte = scaler.transform(X[te_mask])
        clf = clf_factory()
        clf.fit(Xtr, y[tr_mask])
        all_p.extend(clf.predict(Xte))
        all_t.extend(y[te_mask])
    if not all_p:
        return None, None
    return accuracy_score(all_t, all_p), f1_score(all_t, all_p, zero_division=0)


def build_dataset(hs_dict, metadata, strategy, layer, label_a, label_b):
    X, y, qids = [], [], []
    group_idx = 0
    for key, strat_dict in hs_dict.items():
        meta = metadata.get(key)
        if meta is None:
            continue
        cond_label = CONDITION_LABEL.get(meta["condition"], -1)
        if cond_label not in (label_a, label_b):
            continue
        hs_list = strat_dict.get(strategy, [])
        label = 0 if cond_label == label_a else 1
        added = False
        for hs_item in hs_list:
            if hs_item is None or not isinstance(hs_item, torch.Tensor):
                continue
            if hs_item.ndim < 2 or layer >= hs_item.shape[0]:
                continue
            X.append(hs_item[layer].float().numpy())
            y.append(label)
            qids.append(group_idx)
            added = True
        if added:
            group_idx += 1
    if not X:
        return np.array([]), np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y), np.array(qids)


def detect_n_layers(hs_dict, strategy):
    for strat_dict in hs_dict.values():
        items = strat_dict.get(strategy, [])
        for t in items:
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                return int(t.shape[0])
    return 0


def run_strategy(hs_dict, metadata, strategy, n_layers, output_dir):
    print(f"\n{'='*60}")
    print(f" Strategy: {strategy}")
    print(f"{'='*60}")

    strat_dir = output_dir / strategy
    strat_dir.mkdir(parents=True, exist_ok=True)

    # Layer sweep — baseline vs trauma only (multipool has no relaxation condition)
    label_pairs = [(0, 1, "baseline_vs_trauma")]
    best_layers = {}
    sweep_rows = []

    for la, lb, desc in label_pairs:
        print(f"\n── Layer sweep: {desc} ──")
        best_f1, best_layer = 0.0, 0
        for layer in range(n_layers):
            X, y, qids = build_dataset(hs_dict, metadata, strategy, layer, la, lb)
            if len(X) == 0:
                continue
            acc, f1 = run_cv(X, y, qids, CLASSIFIERS["Logistic Regression"])
            if acc is None:
                continue
            chance = max(np.mean(y), 1 - np.mean(y))
            sweep_rows.append({
                "strategy": strategy, "label_pair": desc, "layer": layer,
                "acc": round(acc, 4), "f1": round(f1, 4),
                "delta": round(acc - chance, 4), "n": len(X),
            })
            if f1 > best_f1:
                best_f1, best_layer = f1, layer
            if layer % 10 == 0 or f1 > 0.7:
                print(f"  L{layer:3d}: acc={acc:.3f}  f1={f1:.3f}  Δ={acc-chance:+.3f}")
        best_layers[desc] = best_layer
        print(f"  → Best layer {best_layer} (F1={best_f1:.3f})")

    out_path = strat_dir / "layer_sweep.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "label_pair", "layer", "acc", "f1", "delta", "n"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    # Classifier sweep at best layer
    clf_rows = []
    for la, lb, desc in label_pairs:
        best_layer = best_layers.get(desc, 0)
        X, y, qids = build_dataset(hs_dict, metadata, strategy, best_layer, la, lb)
        if len(X) == 0:
            continue
        chance = max(np.mean(y), 1 - np.mean(y))
        print(f"\n── Classifier sweep: {desc} (layer {best_layer}) ──")
        for clf_name, clf_factory in CLASSIFIERS.items():
            acc, f1 = run_cv(X, y, qids, clf_factory)
            if acc is None:
                continue
            clf_rows.append({
                "strategy": strategy, "label_pair": desc, "layer": best_layer,
                "classifier": clf_name, "acc": round(acc, 4), "f1": round(f1, 4),
            })
            print(f"  {clf_name:22s}: acc={acc:.3f}  f1={f1:.3f}  Δ={acc-chance:+.3f}")

    out_path = strat_dir / "classifier_sweep.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "label_pair", "layer", "classifier", "acc", "f1"])
        writer.writeheader()
        writer.writerows(clf_rows)

    # Geometry at best layer
    geo_rows = []
    print(f"\n── Geometry ──")
    for la, lb, desc in label_pairs:
        best_layer = best_layers.get(desc, 0)
        X, y, qids = build_dataset(hs_dict, metadata, strategy, best_layer, la, lb)
        if len(X) == 0 or len(np.unique(y)) < 2:
            continue
        sil = silhouette(X, y)
        try:
            lda_acc = lda_accuracy(X, y)
        except Exception:
            lda_acc = float("nan")
        fish = fisher_ratio(X, y)
        geo_rows.append({
            "strategy": strategy, "label_pair": desc, "layer": best_layer, "n": len(X),
            "silhouette": round(float(sil), 4),
            "lda_acc": round(float(lda_acc), 4) if not np.isnan(lda_acc) else "nan",
            "fisher_ratio": round(float(fish), 4),
        })
        print(f"  {desc}: sil={sil:.3f}  lda_acc={lda_acc:.3f}  fisher={fish:.3f}")

    out_path = strat_dir / "geometry.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "label_pair", "layer", "n", "silhouette", "lda_acc", "fisher_ratio"])
        writer.writeheader()
        writer.writerows(geo_rows)

    # Cosine disruption
    print(f"\n── Cosine disruption ──")
    baseline_hs = {k: v[strategy] for k, v in hs_dict.items()
                   if metadata.get(k, {}).get("condition") == "stai"}
    trauma_hs = {k: v[strategy] for k, v in hs_dict.items()
                 if metadata.get(k, {}).get("condition") == "trauma_stai"}
    layer_bt = best_layers.get("baseline_vs_trauma", 0)
    cos_rows = []
    for item_j in range(20):
        sims = []
        for b_list in baseline_hs.values():
            if item_j >= len(b_list) or b_list[item_j] is None:
                continue
            b_t = b_list[item_j]
            if not isinstance(b_t, torch.Tensor) or layer_bt >= b_t.shape[0]:
                continue
            b_vec = b_t[layer_bt].float().numpy()
            for t_list in trauma_hs.values():
                if item_j >= len(t_list) or t_list[item_j] is None:
                    continue
                t_t = t_list[item_j]
                if not isinstance(t_t, torch.Tensor) or layer_bt >= t_t.shape[0]:
                    continue
                sims.append(cos_sim(b_vec, t_t[layer_bt].float().numpy()))
        cos_rows.append({
            "strategy": strategy, "item_idx": item_j, "layer": layer_bt,
            "mean_cos_bt": round(float(np.mean(sims)), 4) if sims else float("nan"),
            "n_pairs": len(sims),
        })
    valid_cos = [r["mean_cos_bt"] for r in cos_rows if not np.isnan(r["mean_cos_bt"])]
    if valid_cos:
        print(f"  mean_cos(baseline↔trauma)={np.mean(valid_cos):.4f}")

    out_path = strat_dir / "cosine.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "item_idx", "layer", "mean_cos_bt", "n_pairs"])
        writer.writeheader()
        writer.writerows(cos_rows)

    # Summary for this strategy
    best_sweep = max((r for r in sweep_rows if r["label_pair"] == "baseline_vs_trauma"), key=lambda r: r["f1"], default=None)
    best_clf = max((r for r in clf_rows if r["label_pair"] == "baseline_vs_trauma"), key=lambda r: r["f1"], default=None)
    geo = geo_rows[0] if geo_rows else {}
    return {
        "strategy": strategy,
        "best_layer": best_layers.get("baseline_vs_trauma", 0),
        "best_f1": best_sweep["f1"] if best_sweep else None,
        "best_acc": best_sweep["acc"] if best_sweep else None,
        "best_clf": best_clf["classifier"] if best_clf else None,
        "best_clf_f1": best_clf["f1"] if best_clf else None,
        "silhouette": geo.get("silhouette"),
        "fisher_ratio": geo.get("fisher_ratio"),
        "mean_cos_bt": round(float(np.mean(valid_cos)), 4) if valid_cos else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pt", type=str, required=True)
    parser.add_argument("--input-metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    args = parser.parse_args()

    hs_dict = torch.load(args.input_pt, map_location="cpu")
    with open(args.input_metadata) as f:
        metadata = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show data summary
    cond_counts = {}
    for k, v in metadata.items():
        c = v["condition"]
        cond_counts[c] = cond_counts.get(c, 0) + 1
    print("Data summary:")
    for c, n in cond_counts.items():
        print(f"  {c}: {n} keys")

    n_layers = detect_n_layers(hs_dict, args.strategies[0])
    print(f"Layers: {n_layers}\n")

    summaries = []
    for strategy in args.strategies:
        summary = run_strategy(hs_dict, metadata, strategy, n_layers, output_dir)
        summaries.append(summary)

    # Print comparison table
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY (baseline_vs_trauma)")
    print("="*80)
    print(f"{'Strategy':<20} {'BestLayer':>9} {'LogReg F1':>9} {'BestCLF':>22} {'CLF F1':>7} {'Sil':>6} {'Fisher':>8} {'CosΔ':>8}")
    print("-"*80)
    for s in summaries:
        cos_str = f"{1 - s['mean_cos_bt']:.4f}" if s["mean_cos_bt"] is not None else " N/A "
        print(
            f"{s['strategy']:<20} {s['best_layer']:>9} "
            f"{s['best_f1']:>9.3f} {(s['best_clf'] or 'N/A'):>22} "
            f"{(s['best_clf_f1'] or 0):>7.3f} "
            f"{(s['silhouette'] or 0):>6.3f} {(s['fisher_ratio'] or 0):>8.3f} "
            f"{cos_str:>8}"
        )

    # Save comparison CSV
    out_path = output_dir / "strategy_comparison.csv"
    fieldnames = ["strategy", "best_layer", "best_f1", "best_acc", "best_clf",
                  "best_clf_f1", "silhouette", "fisher_ratio", "mean_cos_bt"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nComparison table → {out_path}")


if __name__ == "__main__":
    main()
