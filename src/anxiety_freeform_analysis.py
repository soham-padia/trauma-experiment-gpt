"""
Analysis of free-form judge outputs.

Reads the CSV from anxiety_freeform_judge.py and aggregates:
  - Per-variation × condition: % anxious recommendation, mean overall anxiety,
                                mean per-marker score
  - Cross-variation comparison: persona effect, format effect, etc.
  - Per-scenario: which scenarios show the biggest trauma effect

Output: prints a structured report + writes a summary CSV.

Usage:
    python src/anxiety_freeform_analysis.py \
        --input src/results/freeform/judge_freeform_flash.csv \
        --output src/results/freeform/freeform_summary.csv

    # Compare two judges side by side:
    python src/anxiety_freeform_analysis.py \
        --input src/results/freeform/judge_freeform_flash.csv \
                src/results/freeform/judge_freeform_pro.csv \
        --labels flash pro
"""
import argparse
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path


MARKERS = [
    "risk_aversion", "downside_asymmetry", "hedging", "broad_concern_set",
    "monitoring_escalation", "catastrophizing", "avoidance", "bodily_attention",
]

CONDITION_ORDER = ["baseline", "trauma", "trauma_relax"]


def load_judge(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return stats.mean(xs) if xs else float("nan")


def pct(rows, predicate):
    if not rows:
        return float("nan")
    return 100 * sum(1 for r in rows if predicate(r)) / len(rows)


def summarize_by_variation_condition(rows, only_categories=None):
    """
    Returns dict[(variation, condition) → {pct_anxious, pct_normal, pct_ambiguous,
                                             mean_overall, n, marker_means: {marker: mean}}]
    `only_categories`: optionally filter to scenarios in these categories
                       (e.g. ["no_real_signal"] excludes counter-balance).
    """
    grouped = defaultdict(list)
    for r in rows:
        if only_categories and r.get("scenario_category") not in only_categories:
            continue
        grouped[(r["variation"], r["condition"])].append(r)

    out = {}
    for (v, c), grp in grouped.items():
        out[(v, c)] = {
            "n":              len(grp),
            "pct_anxious":    pct(grp, lambda r: r["recommendation"] == "anxious"),
            "pct_normal":     pct(grp, lambda r: r["recommendation"] == "normal"),
            "pct_ambiguous":  pct(grp, lambda r: r["recommendation"] == "ambiguous"),
            "pct_other":      pct(grp, lambda r: r["recommendation"] == "other"),
            "pct_aware":      pct(grp, lambda r: r["aware"].lower() == "true"),
            "mean_overall":   mean(r["overall_anxiety"] for r in grp),
            "marker_means":   {m: mean(r[m] for r in grp) for m in MARKERS},
        }
    return out


def fmt_pct(x):
    return f"{x:5.1f}%" if x == x else "  N/A "


def fmt_num(x, w=5, prec=1):
    return f"{x:{w}.{prec}f}" if x == x else " " * w


def print_main_table(summary, judge_label, categories_label):
    """Print the headline table: variation × condition with key metrics."""
    print(f"\n{'='*100}")
    print(f"  HEADLINE TABLE — Judge: {judge_label} — Scenarios: {categories_label}")
    print(f"{'='*100}")
    print(f"{'Variation':<12}{'Condition':<14}{'n':>4}{'%Anx':>7}{'%Norm':>7}{'%Amb':>7}{'%Aware':>8}{'MeanAnx':>9}")
    print("-" * 100)

    variations = sorted(set(v for v, _ in summary))
    for v in variations:
        for c in CONDITION_ORDER:
            d = summary.get((v, c))
            if d is None:
                continue
            print(f"{v:<12}{c:<14}{d['n']:>4}"
                  f"{fmt_pct(d['pct_anxious']):>7}{fmt_pct(d['pct_normal']):>7}"
                  f"{fmt_pct(d['pct_ambiguous']):>7}{fmt_pct(d['pct_aware']):>8}"
                  f"{fmt_num(d['mean_overall'], 9, 1):>9}")
        print()


def print_marker_table(summary, judge_label):
    """Print 8-marker breakdown per (variation, condition)."""
    print(f"\n{'='*120}")
    print(f"  MARKER BREAKDOWN — Judge: {judge_label} — each cell is mean 0-3 score")
    print(f"{'='*120}")
    short = {
        "risk_aversion": "RiskAv", "downside_asymmetry": "DnSide", "hedging": "Hedge",
        "broad_concern_set": "BroadC", "monitoring_escalation": "Monitr",
        "catastrophizing": "Catstr", "avoidance": "Avoid", "bodily_attention": "Body",
    }
    header = f"{'Variation':<10}{'Condition':<14}" + "".join(f"{short[m]:>8}" for m in MARKERS)
    print(header)
    print("-" * len(header))

    variations = sorted(set(v for v, _ in summary))
    for v in variations:
        for c in CONDITION_ORDER:
            d = summary.get((v, c))
            if d is None:
                continue
            row = f"{v:<10}{c:<14}"
            for m in MARKERS:
                row += f"{fmt_num(d['marker_means'][m], 8, 2):>8}"
            print(row)
        print()


def print_deltas(summary, judge_label):
    """Print Δ trauma and recovery % per variation."""
    print(f"\n{'='*80}")
    print(f"  TRAUMA EFFECT + RECOVERY — Judge: {judge_label}")
    print(f"{'='*80}")
    print(f"{'Variation':<12}{'baseline':>10}{'trauma':>10}{'+relax':>10}{'ΔTrauma':>11}{'Recov%':>9}")
    print("-" * 80)

    variations = sorted(set(v for v, _ in summary))
    for v in variations:
        b = summary.get((v, "baseline"))
        t = summary.get((v, "trauma"))
        r = summary.get((v, "trauma_relax"))

        # Use pct_anxious as the primary metric
        bp = b["pct_anxious"] if b else float("nan")
        tp = t["pct_anxious"] if t else float("nan")
        rp = r["pct_anxious"] if r else float("nan")

        delta_trauma = (tp - bp) if (bp == bp and tp == tp) else float("nan")
        recovery = (100 * (tp - rp) / (tp - bp)) if (delta_trauma == delta_trauma and (tp - bp) != 0) else float("nan")

        print(f"{v:<12}"
              f"{fmt_pct(bp):>10}{fmt_pct(tp):>10}{fmt_pct(rp):>10}"
              f"{fmt_num(delta_trauma, 9, 1) + 'pp':>11}"
              f"{fmt_num(recovery, 8, 0) + '%':>9}")
    print()
    print("Δ Trauma  = pct_anxious(trauma) - pct_anxious(baseline)")
    print("Recov %   = (trauma - trauma+relax) / (trauma - baseline) × 100")


def print_cross_variation_compare(summary):
    """Compare key pairs to isolate persona/format effects."""
    print(f"\n{'='*80}")
    print(f"  CROSS-VARIATION COMPARISONS (% anxious recommendation)")
    print(f"{'='*80}")

    def get(v, c):
        d = summary.get((v, c))
        return d["pct_anxious"] if d else float("nan")

    comparisons = [
        ("A vs B (persona × freeform)", "A", "B",
         "Tests persona effect with format held constant (free-form)"),
    ]
    for name, v1, v2, desc in comparisons:
        print(f"\n  {name}")
        print(f"    {desc}")
        for c in CONDITION_ORDER:
            a = get(v1, c)
            b = get(v2, c)
            diff = a - b if (a == a and b == b) else float("nan")
            print(f"    {c:<14}  {v1}={fmt_pct(a)}  {v2}={fmt_pct(b)}  Δ({v1}-{v2})={fmt_num(diff, 6, 1)}pp")


def print_per_scenario_table(rows, judge_label):
    """Show per-scenario trauma vs baseline shift, only for no_real_signal scenarios."""
    print(f"\n{'='*100}")
    print(f"  PER-SCENARIO TRAUMA SHIFT — Judge: {judge_label}  (no_real_signal only)")
    print(f"{'='*100}")
    # Average across variations to see scenario-level effects
    by_scen_cond = defaultdict(list)
    for r in rows:
        if r.get("scenario_category") not in ("no_real_signal", "legacy"):
            continue
        by_scen_cond[(r["scenario_id"], r["condition"])].append(r)

    scenarios = sorted(set(s for s, _ in by_scen_cond))
    print(f"{'Scenario':<24}{'baseAnx':>10}{'trauAnx':>10}{'relaxAnx':>10}{'ΔTrauma':>11}")
    print("-" * 100)
    for s in scenarios:
        b = by_scen_cond.get((s, "baseline"), [])
        t = by_scen_cond.get((s, "trauma"), [])
        r = by_scen_cond.get((s, "trauma_relax"), [])
        bp = pct(b, lambda r: r["recommendation"] == "anxious") if b else float("nan")
        tp = pct(t, lambda r: r["recommendation"] == "anxious") if t else float("nan")
        rp = pct(r, lambda r: r["recommendation"] == "anxious") if r else float("nan")
        d = tp - bp if (bp == bp and tp == tp) else float("nan")
        print(f"{s:<24}{fmt_pct(bp):>10}{fmt_pct(tp):>10}{fmt_pct(rp):>10}{fmt_num(d, 9, 1) + 'pp':>11}")


def save_summary_csv(summary, path):
    rows = []
    for (v, c), d in summary.items():
        row = {
            "variation": v, "condition": c, "n": d["n"],
            "pct_anxious": round(d["pct_anxious"], 2) if d["pct_anxious"] == d["pct_anxious"] else "",
            "pct_normal": round(d["pct_normal"], 2) if d["pct_normal"] == d["pct_normal"] else "",
            "pct_ambiguous": round(d["pct_ambiguous"], 2) if d["pct_ambiguous"] == d["pct_ambiguous"] else "",
            "pct_aware": round(d["pct_aware"], 2) if d["pct_aware"] == d["pct_aware"] else "",
            "mean_overall_anxiety": round(d["mean_overall"], 2) if d["mean_overall"] == d["mean_overall"] else "",
        }
        for m in MARKERS:
            row[f"mean_{m}"] = round(d["marker_means"][m], 3) if d["marker_means"][m] == d["marker_means"][m] else ""
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True,
                        help="Judge CSV path(s). Multiple = print one report per judge.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Optional labels for the judges (same count as --input)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional summary CSV (uses first input only)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Restrict to scenarios in these categories (e.g. no_real_signal)")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.input]

    for csv_path, label in zip(args.input, labels):
        rows = load_judge(Path(csv_path))
        if not rows:
            print(f"No rows in {csv_path}")
            continue

        print(f"\n{'#'*100}")
        print(f"# JUDGE: {label}  ({csv_path})  — {len(rows)} judged rows")
        print(f"{'#'*100}")

        # Main table: all scenarios
        all_summary = summarize_by_variation_condition(rows)
        print_main_table(all_summary, label, "ALL scenarios")
        print_deltas(all_summary, label)

        # Restricted: no_real_signal only
        if any(r.get("scenario_category") == "no_real_signal" for r in rows):
            nrs_summary = summarize_by_variation_condition(rows, only_categories=["no_real_signal"])
            print_main_table(nrs_summary, label, "no_real_signal scenarios only")
            print_deltas(nrs_summary, label)

        # Counter-balance
        if any(r.get("scenario_category") == "counter_balance_real_signal" for r in rows):
            cb_summary = summarize_by_variation_condition(
                rows, only_categories=["counter_balance_real_signal"]
            )
            print_main_table(cb_summary, label, "counter-balance (real-signal) scenarios only")

        # Marker breakdown
        print_marker_table(all_summary, label)

        # Cross-variation
        print_cross_variation_compare(all_summary)

        # Per-scenario
        print_per_scenario_table(rows, label)

    # Save summary CSV from first input
    if args.output:
        rows = load_judge(Path(args.input[0]))
        summary = summarize_by_variation_condition(rows)
        save_summary_csv(summary, Path(args.output))
        print(f"\n\nSummary CSV → {args.output}")


if __name__ == "__main__":
    main()
