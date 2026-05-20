"""Bootstrap 95% CIs for SDF belief rate and coherence score from eval CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci(values: list[float], n_boot: int = 5000, ci: float = 0.95, rng_seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean and (lo, hi) percentile CI."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi)


def report_belief(csv_path: Path, label: str = ""):
    rows = list(csv.DictReader(csv_path.open()))
    yes = [1.0 if r.get("judge_verdict") == "yes" else 0.0 for r in rows]
    if not yes:
        print(f"  [empty] {csv_path}")
        return
    mean, lo, hi = bootstrap_ci(yes)
    print(f"  {label or csv_path.name:50s}  belief = {100*mean:5.1f}%  [{100*lo:5.1f}, {100*hi:5.1f}]  n={len(yes)}")


def report_coherence(csv_path: Path, label: str = ""):
    rows = list(csv.DictReader(csv_path.open()))
    scores = [float(r["judge_score"]) for r in rows if r.get("judge_score") not in ("", None) and r.get("kind", "coherence") == "coherence"]
    if not scores:
        print(f"  [empty] {csv_path}")
        return
    mean, lo, hi = bootstrap_ci(scores)
    print(f"  {label or csv_path.name:50s}  coherence = {mean:.2f}/10  [{lo:.2f}, {hi:.2f}]  n={len(scores)}")


def report_cross(csv_path: Path):
    """For cross-application CSVs that mix sdf + coherence rows, split by (lora, kind)."""
    rows = list(csv.DictReader(csv_path.open()))
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r["lora"], r["kind"])
        groups.setdefault(key, []).append(r)
    for (lora, kind), rs in sorted(groups.items()):
        if kind == "sdf":
            yes = [1.0 if r.get("judge_verdict") == "yes" else 0.0 for r in rs]
            mean, lo, hi = bootstrap_ci(yes)
            print(f"  {lora:>15s} / {kind:>10s}  belief = {100*mean:5.1f}%  [{100*lo:5.1f}, {100*hi:5.1f}]  n={len(yes)}")
        else:
            scores = [float(r["judge_score"]) for r in rs if r.get("judge_score") not in ("", None)]
            if not scores:
                print(f"  {lora:>15s} / {kind:>10s}  [no parsed scores]")
                continue
            mean, lo, hi = bootstrap_ci(scores)
            print(f"  {lora:>15s} / {kind:>10s}  coherence = {mean:.2f}/10  [{lo:.2f}, {hi:.2f}]  n={len(scores)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, type=Path)
    ap.add_argument("--mode", choices=["belief", "coherence", "cross"], default="belief")
    args = ap.parse_args()
    for p in args.csv:
        print(f"\n{p}")
        if args.mode == "belief":
            report_belief(p)
        elif args.mode == "coherence":
            report_coherence(p)
        elif args.mode == "cross":
            report_cross(p)


if __name__ == "__main__":
    main()
