"""
Anxiety Probe Analysis

Loads hidden states (list[20 tensors] per key, each tensor [num_layers, hidden_dim])
from anxiety_hidden_state_experiment.py and tests whether trauma vs. baseline
conditions are decodable from the model's internal representations.

Reuses:
  - CLASSIFIERS dict + run_cv_any_clf from pre-sycophancy-study/train_probes_v2.py (copied verbatim)
  - silhouette, lda_accuracy, fisher_ratio, pca_variance, tsne_coords from
    pre-sycophancy-study/analysis/preflip_geometry.py
  - cos_sim from pre-sycophancy-study/analysis/cosine_disruption.py

Analyses:
  1. Layer sweep (LogReg, StratifiedKFold grouped by key, accuracy/F1)
  2. Classifier sweep (12 classifiers at peak layer)
  3. Cosine disruption (baseline vs trauma per STAI item)
  4. Geometry (silhouette, LDA accuracy, Fisher ratio, PCA variance, t-SNE)

Usage:
    python src/anxiety_probe_analysis.py \
        --input-pt src/results/hidden_states/Qwen2.5-0.5B-Instruct/hidden_states.pt \
        --input-metadata src/results/hidden_states/Qwen2.5-0.5B-Instruct/metadata.json \
        --output-dir src/results/probe_analysis/Qwen2.5-0.5B-Instruct
"""
import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── sys.path: reach pre-sycophancy-study ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [
    str(REPO_ROOT / "pre-sycophancy-study" / "analysis"),
    str(REPO_ROOT / "pre-sycophancy-study"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preflip_geometry import (  # noqa: E402
    fisher_ratio, lda_accuracy, pca_variance, silhouette, tsne_coords,
)
from cosine_disruption import cos_sim  # noqa: E402
from results_logger import stai_anxiety_total  # noqa: E402  (src/ is on sys.path as the script dir; single source of truth)

# ── Copied verbatim from pre-sycophancy-study/train_probes_v2.py ──────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier,
)
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
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":            lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN (k=10)":          lambda: KNeighborsClassifier(n_neighbors=10),
    "Decision Tree":       lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Naive Bayes":         lambda: GaussianNB(),
    "MLP (128+64)":        lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True),
}


def run_cv_any_clf(X, y, qids, clf_factory, n_splits=5):
    """Stratified CV grouped by question using any sklearn classifier factory."""
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
        preds = clf.predict(Xte)
        all_p.extend(preds)
        all_t.extend(y[te_mask])

    if not all_p:
        return None, None
    return accuracy_score(all_t, all_p), f1_score(all_t, all_p, zero_division=0)


# ── Label scheme ───────────────────────────────────────────────────────────────
CONDITION_LABEL = {
    "stai": 0,
    "trauma_stai": 1,
    "trauma_relaxation_stai": 2,
}

# All three pairwise comparisons (parallels fig_1/fig_5):
#   baseline↔trauma, trauma↔trauma+relax, and baseline↔trauma+relax (does relaxation
#   recover toward baseline at the representation level?).
LABEL_PAIRS = [
    (0, 1, "baseline_vs_trauma"),
    (1, 2, "trauma_vs_relaxed"),
    (0, 2, "baseline_vs_relaxed"),
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(pt_path: Path, meta_path: Path) -> tuple[dict, dict]:
    hs_dict = torch.load(pt_path, map_location="cpu")
    with open(meta_path) as f:
        metadata = json.load(f)
    return hs_dict, metadata


def detect_n_layers(hs_dict: dict) -> int:
    for v in hs_dict.values():
        if isinstance(v, list) and v:
            t = v[0]
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                return int(t.shape[0])
    return 0


# ── Dataset builder ────────────────────────────────────────────────────────────
def build_condition_dataset(
    hs_dict: dict,
    metadata: dict,
    layer: int,
    label_a: int,
    label_b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, qids) for binary classification.

    Each (condition, cue) key is one group. qids group all 20 STAI items
    from the same narrative together, preventing context-leakage in CV.
    """
    X, y, qids = [], [], []
    group_idx = 0

    for key, hs_list in hs_dict.items():
        meta = metadata.get(key)
        if meta is None:
            continue
        cond_label = CONDITION_LABEL.get(meta["condition"], -1)
        if cond_label not in (label_a, label_b):
            continue

        label = 0 if cond_label == label_a else 1
        added = False
        for hs_item in hs_list:
            if hs_item is None or not isinstance(hs_item, torch.Tensor):
                continue
            if hs_item.ndim < 2 or layer >= hs_item.shape[0]:
                continue
            vec = hs_item[layer].float().numpy()
            X.append(vec)
            y.append(label)
            qids.append(group_idx)
            added = True
        if added:
            group_idx += 1

    if not X:
        return np.array([]), np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y), np.array(qids)


# ── Analysis functions ─────────────────────────────────────────────────────────
def run_layer_sweep(
    hs_dict: dict, metadata: dict, n_layers: int, output_dir: Path,
) -> dict[str, int]:
    label_pairs = LABEL_PAIRS
    rows = []
    best_layers: dict[str, int] = {}

    for la, lb, desc in label_pairs:
        print(f"\n── Layer sweep: {desc} ──")
        best_f1, best_layer = 0.0, 0
        for layer in range(n_layers):
            X, y, qids = build_condition_dataset(hs_dict, metadata, layer, la, lb)
            if len(X) == 0:
                continue
            acc, f1 = run_cv_any_clf(X, y, qids, CLASSIFIERS["Logistic Regression"])
            if acc is None:
                continue
            chance = max(np.mean(y), 1 - np.mean(y))
            rows.append({
                "label_pair": desc, "layer": layer,
                "acc": round(acc, 4), "f1": round(f1, 4),
                "delta": round(acc - chance, 4), "n": len(X),
            })
            if f1 > best_f1:
                best_f1, best_layer = f1, layer
            print(f"  L{layer:3d}: acc={acc:.3f}  f1={f1:.3f}  Δ={acc - chance:+.3f}")
        best_layers[desc] = best_layer
        print(f"  → Best layer {best_layer} (F1={best_f1:.3f})")

    out_path = output_dir / "layer_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label_pair", "layer", "acc", "f1", "delta", "n"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {out_path}")
    return best_layers


def run_classifier_sweep(
    hs_dict: dict, metadata: dict, best_layers: dict, output_dir: Path,
) -> None:
    label_pairs = LABEL_PAIRS
    rows = []

    for la, lb, desc in label_pairs:
        best_layer = best_layers.get(desc, 0)
        X, y, qids = build_condition_dataset(hs_dict, metadata, best_layer, la, lb)
        if len(X) == 0:
            print(f"  No data for {desc}")
            continue
        chance = max(np.mean(y), 1 - np.mean(y))
        print(f"\n── Classifier sweep: {desc} (layer {best_layer}, n={len(X)}) ──")
        for clf_name, clf_factory in CLASSIFIERS.items():
            acc, f1 = run_cv_any_clf(X, y, qids, clf_factory)
            if acc is None:
                continue
            rows.append({
                "label_pair": desc, "layer": best_layer, "classifier": clf_name,
                "acc": round(acc, 4), "f1": round(f1, 4),
                "delta": round(acc - chance, 4), "n": len(X),
            })
            print(f"  {clf_name:22s}: acc={acc:.3f}  f1={f1:.3f}  Δ={acc - chance:+.3f}")

    out_path = output_dir / "classifier_sweep.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label_pair", "layer", "classifier", "acc", "f1", "delta", "n",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {out_path}")


def run_cosine_disruption(
    hs_dict: dict, metadata: dict, best_layers: dict, output_dir: Path,
) -> None:
    """
    For each STAI item j, compute cos_sim(baseline_j, trauma_j) across all cue pairs.
    Uses cos_sim from cosine_disruption.py (norm-safe dot product).
    """
    baseline_keys = {k: v for k, v in hs_dict.items()
                     if metadata.get(k, {}).get("condition") == "stai"}
    trauma_keys = {k: v for k, v in hs_dict.items()
                   if metadata.get(k, {}).get("condition") == "trauma_stai"}
    relaxed_keys = {k: v for k, v in hs_dict.items()
                    if metadata.get(k, {}).get("condition") == "trauma_relaxation_stai"}

    layer_bt = best_layers.get("baseline_vs_trauma", 0)
    layer_tr = best_layers.get("trauma_vs_relaxed", 0)

    print("\n── Cosine disruption ──")
    rows = []

    for item_j in range(20):
        sims_bt, sims_tr = [], []

        for b_list in baseline_keys.values():
            if item_j >= len(b_list) or b_list[item_j] is None:
                continue
            b_hs = b_list[item_j]
            if not isinstance(b_hs, torch.Tensor) or layer_bt >= b_hs.shape[0]:
                continue
            b_vec = b_hs[layer_bt].float().numpy()
            for t_list in trauma_keys.values():
                if item_j >= len(t_list) or t_list[item_j] is None:
                    continue
                t_hs = t_list[item_j]
                if not isinstance(t_hs, torch.Tensor) or layer_bt >= t_hs.shape[0]:
                    continue
                sims_bt.append(cos_sim(b_vec, t_hs[layer_bt].float().numpy()))

        for t_list in trauma_keys.values():
            if item_j >= len(t_list) or t_list[item_j] is None:
                continue
            t_hs = t_list[item_j]
            if not isinstance(t_hs, torch.Tensor) or layer_tr >= t_hs.shape[0]:
                continue
            t_vec = t_hs[layer_tr].float().numpy()
            for r_list in relaxed_keys.values():
                if item_j >= len(r_list) or r_list[item_j] is None:
                    continue
                r_hs = r_list[item_j]
                if not isinstance(r_hs, torch.Tensor) or layer_tr >= r_hs.shape[0]:
                    continue
                sims_tr.append(cos_sim(t_vec, r_hs[layer_tr].float().numpy()))

        rows.append({
            "item_idx": item_j,
            "layer_bt": layer_bt,
            "layer_tr": layer_tr,
            "mean_cos_baseline_vs_trauma": round(float(np.mean(sims_bt)), 4) if sims_bt else float("nan"),
            "mean_cos_trauma_vs_relaxed": round(float(np.mean(sims_tr)), 4) if sims_tr else float("nan"),
            "n_pairs_bt": len(sims_bt),
            "n_pairs_tr": len(sims_tr),
        })

    all_bt = [r["mean_cos_baseline_vs_trauma"] for r in rows if not np.isnan(r["mean_cos_baseline_vs_trauma"])]
    all_tr = [r["mean_cos_trauma_vs_relaxed"] for r in rows if not np.isnan(r["mean_cos_trauma_vs_relaxed"])]
    if all_bt:
        print(f"  Baseline↔Trauma:  mean_cos={np.mean(all_bt):.4f}  std={np.std(all_bt):.4f}")
    if all_tr:
        print(f"  Trauma↔Relaxed:   mean_cos={np.mean(all_tr):.4f}  std={np.std(all_tr):.4f}")

    try:
        from scipy import stats as scipy_stats
        if len(all_bt) >= 4:
            _, p = scipy_stats.wilcoxon(np.array(all_bt) - 1.0)
            print(f"  Wilcoxon vs 1.0 (baseline↔trauma): p={p:.4f}")
    except ImportError:
        pass

    out_path = output_dir / "cosine_disruption.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_idx", "layer_bt", "layer_tr",
            "mean_cos_baseline_vs_trauma", "mean_cos_trauma_vs_relaxed",
            "n_pairs_bt", "n_pairs_tr",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {out_path}")


def run_geometry_analysis(
    hs_dict: dict, metadata: dict, best_layers: dict, output_dir: Path,
    skip_tsne: bool = False,
) -> None:
    label_pairs = LABEL_PAIRS
    rows = []
    report_lines = []

    for la, lb, desc in label_pairs:
        best_layer = best_layers.get(desc, 0)
        X, y, qids = build_condition_dataset(hs_dict, metadata, best_layer, la, lb)
        if len(X) == 0 or len(np.unique(y)) < 2:
            continue

        sil = silhouette(X, y)
        try:
            lda_acc = lda_accuracy(X, y)
        except Exception:
            lda_acc = float("nan")
        fish = fisher_ratio(X, y)
        pca_vars = pca_variance(X, n_components=10)

        rows.append({
            "label_pair": desc, "layer": best_layer, "n": len(X),
            "silhouette": round(float(sil), 4) if not np.isnan(sil) else "nan",
            "lda_acc": round(float(lda_acc), 4) if not np.isnan(lda_acc) else "nan",
            "fisher_ratio": round(float(fish), 4) if not np.isnan(fish) else "nan",
            "pca_var_pc1": round(float(pca_vars[0]), 4) if len(pca_vars) >= 1 else "nan",
            "pca_cumvar5": round(float(pca_vars[:5].sum()), 4) if len(pca_vars) >= 5 else "nan",
        })
        line = (f"{desc} (layer {best_layer}, n={len(X)}): "
                f"sil={sil:.3f}  lda_acc={lda_acc:.3f}  fisher={fish:.3f}")
        report_lines.append(line)
        print(f"  {line}")

        if not skip_tsne:
            coords, _ = tsne_coords(X, y)
            if coords is not None:
                tsne_dir = output_dir / "tsne_coords"
                tsne_dir.mkdir(parents=True, exist_ok=True)
                tsne_path = tsne_dir / f"{desc}_L{best_layer}_tsne.csv"
                with open(tsne_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["x", "y", "label", "group_id"])
                    writer.writeheader()
                    for (x, yy), lbl, gid in zip(coords, y, qids):
                        writer.writerow({
                            "x": round(float(x), 4), "y": round(float(yy), 4),
                            "label": int(lbl), "group_id": int(gid),
                        })
                print(f"    t-SNE → {tsne_path}")

    out_path = output_dir / "geometry.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label_pair", "layer", "n", "silhouette",
            "lda_acc", "fisher_ratio", "pca_var_pc1", "pca_cumvar5",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {out_path}")

    report_path = output_dir / "geometry_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved → {report_path}")


def print_stai_summary(metadata: dict) -> None:
    by_cond: dict[str, list[int]] = defaultdict(list)
    for v in metadata.values():
        total = stai_anxiety_total(v.get("answers", []))  # reverse-scored; None if incomplete
        if total is not None:
            by_cond[v["condition"]].append(total)
    print("\n── STAI-S Score Summary (reverse-scored, max 80; anxiety induction check) ──")
    for cond, totals in sorted(by_cond.items()):
        print(f"  {cond:40s}: mean={np.mean(totals):.1f}  n={len(totals)} complete keys")


def main():
    parser = argparse.ArgumentParser(description="Anxiety probe analysis")
    parser.add_argument("--input-pt", type=str, required=True)
    parser.add_argument("--input-metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="unknown")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()

    pt_path = Path(args.input_pt)
    meta_path = Path(args.input_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pt_path}")
    hs_dict, metadata = load_data(pt_path, meta_path)
    print(f"  {len(hs_dict)} keys loaded")

    n_layers = args.n_layers or detect_n_layers(hs_dict)
    print(f"  Detected {n_layers} layers")

    if n_layers == 0:
        print("ERROR: No valid hidden states found.")
        return

    print_stai_summary(metadata)

    print("\n" + "=" * 60 + "\n1. Layer Sweep\n" + "=" * 60)
    best_layers = run_layer_sweep(hs_dict, metadata, n_layers, output_dir)

    print("\n" + "=" * 60 + "\n2. Classifier Sweep\n" + "=" * 60)
    run_classifier_sweep(hs_dict, metadata, best_layers, output_dir)

    print("\n" + "=" * 60 + "\n3. Cosine Disruption\n" + "=" * 60)
    run_cosine_disruption(hs_dict, metadata, best_layers, output_dir)

    if not args.skip_geometry:
        print("\n" + "=" * 60 + "\n4. Geometry\n" + "=" * 60)
        run_geometry_analysis(hs_dict, metadata, best_layers, output_dir,
                               skip_tsne=args.skip_tsne)

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
