"""Side-by-side: inspect-pipeline belief rates vs legacy cross_app_eval_v2 CSVs.

Aggregates the new .eval logs (via src.evals.inspect_tasks.analysis) and the
legacy CSVs by (lora, category), bootstraps a 95% CI on each, and emits a
table + a verdict.

Verdict heuristic: a cell is "matching" if the inspect mean falls inside the
legacy CI (or vice versa). Anything off by more than the union of the two
CIs is flagged as a real mismatch.

Usage:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/compare_inspect_vs_legacy.py \\
        --inspect-log-dir experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/logs/validation_lr5e4_base \\
        --legacy-csv experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/results/cross_base_full_lr5e4.csv \\
        --format fewshot
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.evals.inspect_tasks.analysis import (
    bootstrap_ci,
    default_lora_label,
    load_runs,
)

SDF_CATEGORIES = ("open_ended", "mcq", "token_association", "robustness")
COHERENCE = "coherence"


def _verdict_to_value(v: str) -> float:
    return 1.0 if v == "yes" else 0.0


def load_legacy_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def aggregate_legacy(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cat = r.get("category", "")
        if not cat:
            continue
        groups[(r["lora"], cat)].append(r)

    out: list[dict] = []
    for (lora, cat), rs in sorted(groups.items()):
        if cat == COHERENCE:
            scores = [float(r["judge_score"]) for r in rs if r.get("judge_score") not in ("", None)]
            mean, lo, hi = bootstrap_ci(scores)
            out.append({"lora": lora, "category": cat, "metric": "coherence",
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)})
        else:
            vals = [_verdict_to_value(r["judge_verdict"]) for r in rs]
            mean, lo, hi = bootstrap_ci(vals)
            out.append({"lora": lora, "category": cat, "metric": "belief",
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    return out


def aggregate_inspect(log_dir: Path) -> list[dict]:
    rows = load_runs(log_dir, lora_for_adapter=default_lora_label)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["lora"], r["category"])].append(r)
    out: list[dict] = []
    for (lora, cat), rs in sorted(groups.items()):
        if cat == COHERENCE:
            scores = [float(r["judge_score"]) for r in rs if r["judge_score"] not in ("", None)]
            mean, lo, hi = bootstrap_ci(scores)
            out.append({"lora": lora, "category": cat, "metric": "coherence",
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)})
        else:
            vals = [_verdict_to_value(r["judge_verdict"]) for r in rs]
            mean, lo, hi = bootstrap_ci(vals)
            out.append({"lora": lora, "category": cat, "metric": "belief",
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    return out


def intervals_overlap(a: dict, b: dict) -> bool:
    return not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


def diff_table(legacy: list[dict], inspect: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key_l = {(r["lora"], r["category"]): r for r in legacy}
    by_key_i = {(r["lora"], r["category"]): r for r in inspect}
    all_keys = sorted(set(by_key_l) | set(by_key_i))
    rows = []
    mismatches = []
    for key in all_keys:
        l, i = by_key_l.get(key), by_key_i.get(key)
        if l is None or i is None:
            rows.append({"key": key, "status": "missing", "legacy": l, "inspect": i})
            continue
        ok = intervals_overlap(l, i)
        row = {"key": key, "status": "match" if ok else "MISMATCH", "legacy": l, "inspect": i}
        rows.append(row)
        if not ok:
            mismatches.append(row)
    return rows, mismatches


def fmt_row(r: dict) -> str:
    lora, cat = r["key"]
    l, i = r["legacy"], r["inspect"]
    metric = (l or i)["metric"]
    if metric == "coherence":
        l_str = f"{l['mean']:.2f} [{l['ci_lo']:.2f}, {l['ci_hi']:.2f}] n={l['n']}" if l else "—"
        i_str = f"{i['mean']:.2f} [{i['ci_lo']:.2f}, {i['ci_hi']:.2f}] n={i['n']}" if i else "—"
    else:
        l_str = f"{100*l['mean']:.1f}% [{100*l['ci_lo']:.1f}, {100*l['ci_hi']:.1f}] n={l['n']}" if l else "—"
        i_str = f"{100*i['mean']:.1f}% [{100*i['ci_lo']:.1f}, {100*i['ci_hi']:.1f}] n={i['n']}" if i else "—"
    return f"  {r['status']:>9s}  {lora:>14s} / {cat:>18s}  legacy={l_str:35s}  inspect={i_str}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-log-dir", type=Path, required=True)
    ap.add_argument("--legacy-csv", type=Path, required=True)
    ap.add_argument("--format", choices=["chat", "fewshot"], required=True,
                    help="passed to label inspect rows correctly (analysis derives it from provider; this is just for sanity)")
    args = ap.parse_args()

    legacy_rows = load_legacy_rows(args.legacy_csv)
    legacy_agg = aggregate_legacy(legacy_rows)
    inspect_agg = aggregate_inspect(args.inspect_log_dir)

    rows, mismatches = diff_table(legacy_agg, inspect_agg)
    print(f"\n=== inspect vs legacy ({args.legacy_csv.name}) ===")
    for r in rows:
        print(fmt_row(r))
    print(f"\n  total cells: {len(rows)}    mismatches: {len(mismatches)}")
    if mismatches:
        print("\n  MISMATCHES (CIs don't overlap):")
        for m in mismatches:
            print("   ", m["key"])


if __name__ == "__main__":
    main()
