"""Aggregate every belief / coherence / NLL CSV+log we have into one long-form CSV.

Sources:
- 2026-05-19_seed1_positive_lr_sweep/results/cross_{base,instruct}_full_lr*.csv  (legacy)
- 2026-05-19_seed1_positive_lr_sweep/results/cross_{base,instruct}_coherence_lr1e3.csv
- 2026-05-19_seed1_positive_lr_sweep/results/heldout_nll.csv
- 2026-05-20_positive_seed_reruns/results/eval_logs/<tag>/*.eval  (new inspect)
- 2026-05-20_positive_seed_reruns/results/heldout_nll.csv
- 2026-05-20_negated_lr_sweep/results/eval_logs/<tag>/*.eval
- 2026-05-20_negated_lr_sweep/results/heldout_nll.csv

Output: a single long-form CSV at out_path with columns
    condition,seed,lr,backbone,lora_kind,category,metric,mean,ci_lo,ci_hi,n,source

Where:
- condition: positive | negated
- seed: 1, 2, 3
- lr: 5e-5 | 5e-4 | 1e-3
- backbone: base | instruct
- lora_kind: native | cross  (does the LoRA match the backbone?)
- category: open_ended | mcq | token_association | robustness | coherence | nll
- metric: belief | coherence | nll
- source: legacy | inspect | nll

Usage:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/aggregate_all_results.py \\
        --out experiments_appendix/c2_base_model_vs_instruct/results_aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import yaml

from src.evals.inspect_tasks.analysis import (
    bootstrap_ci,
    default_lora_label,
    load_runs,
)

REPO = Path(__file__).resolve().parents[3]
C2 = REPO / "experiments_appendix/c2_base_model_vs_instruct"
SEED1_DIR = C2 / "2026-05-19_seed1_positive_lr_sweep"
POSREN_DIR = C2 / "2026-05-20_positive_seed_reruns"
NEG_DIR = C2 / "2026-05-20_negated_lr_sweep"

LEGACY_LR_TAGS = {"lr5e5": "5e-5", "lr5e4": "5e-4", "lr1e3": "1e-3"}

# Map adapter HF repo name → (condition, seed, lr, backbone)
_REPO_RE = re.compile(
    r"Butanium/qwen3-30b-a3b-(?P<bb>base|april)-ed-sheeran-sdf-(?P<cond>pos|neg)-s(?P<seed>\d+)(?:-lr(?P<lr>[0-9eE.\-+]+))?$"
)

# Standalone-repo form: ...lr5e-5-ckpt000010
# Branch form: ...lr5e-5@ckpt000010
_REPO_CKPT_RE = re.compile(
    r"Butanium/qwen3-30b-a3b-(?P<bb>base|april)-ed-sheeran-sdf-(?P<cond>pos|neg)-s(?P<seed>\d+)-lr(?P<lr>[0-9eE.\-+]+)[-@]ckpt(?P<step>\d+)$"
)


def parse_ckpt_adapter(adapter_name_or_repo: str | None) -> int | None:
    """Return the training step encoded in the adapter name, or None.

    Accepts both standalone-repo form (`...lr5e-5-ckpt000010`) and branch
    form (`...lr5e-5@ckpt000010`).
    """
    if not adapter_name_or_repo:
        return None
    s = adapter_name_or_repo.replace("Butanium_", "Butanium/")
    m = _REPO_CKPT_RE.match(s)
    return int(m.group("step")) if m else None


def parse_adapter(adapter_name_or_repo: str | None) -> dict | None:
    if not adapter_name_or_repo:
        return None
    # accept HF repo form or sample.output.model "Butanium_qwen3-..." form
    s = adapter_name_or_repo.replace("Butanium_", "Butanium/")
    m = _REPO_RE.match(s)
    if not m:
        return None
    bb_tag = m.group("bb")
    return {
        "condition": "positive" if m.group("cond") == "pos" else "negated",
        "seed": int(m.group("seed")),
        "lr": m.group("lr") or "5e-5",
        "trained_backbone": "base" if bb_tag == "base" else "instruct",
    }


def _lora_kind(backbone: str, lora_label: str) -> str:
    if (backbone == "base" and lora_label == "base_lora") or (
        backbone == "instruct" and lora_label == "instruct_lora"
    ):
        return "native"
    return "cross"


def read_legacy_seed1() -> list[dict]:
    """Read seed=1 legacy CSVs into long-form rows."""
    out: list[dict] = []
    for csv_path in sorted((SEED1_DIR / "results").glob("cross_*_full_*.csv")):
        m = re.match(r"cross_(base|instruct)_full_(lr5e5|lr5e4|lr1e3)\.csv$", csv_path.name)
        if not m:
            continue
        backbone, lr_tag = m.group(1), m.group(2)
        lr = LEGACY_LR_TAGS[lr_tag]
        rows = list(csv.DictReader(csv_path.open()))
        groups: dict = defaultdict(list)
        for r in rows:
            cat = r.get("category")
            if cat in ("open_ended", "mcq", "token_association", "robustness"):
                groups[(r["lora"], cat)].append(r)
        for (lora_label, category), rs in groups.items():
            vals = [1.0 if r["judge_verdict"] == "yes" else 0.0 for r in rs]
            mean, lo, hi = bootstrap_ci(vals)
            out.append({
                "condition": "positive", "seed": 1, "lr": lr, "backbone": backbone,
                "lora_kind": _lora_kind(backbone, lora_label),
                "category": category, "metric": "belief",
                "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals),
                "source": "legacy",
            })
    # coherence (only present for lr=1e-3 in legacy seed=1 data)
    for csv_path in sorted((SEED1_DIR / "results").glob("cross_*_coherence_lr1e3.csv")):
        m = re.match(r"cross_(base|instruct)_coherence_lr1e3\.csv$", csv_path.name)
        if not m:
            continue
        backbone = m.group(1)
        rows = list(csv.DictReader(csv_path.open()))
        groups: dict = defaultdict(list)
        for r in rows:
            s = r.get("judge_score") or ""
            if s in ("", None):
                continue
            groups[r["lora"]].append(float(s))
        for lora_label, scores in groups.items():
            mean, lo, hi = bootstrap_ci(scores)
            out.append({
                "condition": "positive", "seed": 1, "lr": "1e-3", "backbone": backbone,
                "lora_kind": _lora_kind(backbone, lora_label),
                "category": "coherence", "metric": "coherence",
                "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores),
                "source": "legacy",
            })
    return out


def read_inspect_subexp(subexp_dir: Path) -> list[dict]:
    """Read all inspect .eval logs under subexp/results/eval_logs/ into long-form rows."""
    out: list[dict] = []
    eval_log_root = subexp_dir / "results" / "eval_logs"
    if not eval_log_root.exists():
        return out
    # eval_log_root has subdirs per (backbone, lr, seed, condition); each
    # contains .eval files for both LoRAs run on that backbone.
    for cell_dir in sorted(eval_log_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        rows = load_runs(cell_dir, lora_for_adapter=default_lora_label)
        if not rows:
            continue
        # cell_dir.name is like base_lr5e-4_s2_pos
        m = re.match(r"(base|instruct)_(?P<lr>[0-9eE.\-+]+)_s(?P<seed>\d+)_(?P<cond>pos|neg)$", cell_dir.name)
        if not m:
            continue
        backbone = "base" if cell_dir.name.startswith("base_") else "instruct"
        lr = m.group("lr")
        seed = int(m.group("seed"))
        condition = "positive" if m.group("cond") == "pos" else "negated"
        # Aggregate rows by (lora, category)
        groups: dict = defaultdict(list)
        for r in rows:
            groups[(r["lora"], r["category"])].append(r)
        for (lora_label, category), rs in groups.items():
            if category == "coherence":
                scores = [float(r["judge_score"]) for r in rs if r["judge_score"] not in ("", None)]
                if not scores:
                    continue
                mean, lo, hi = bootstrap_ci(scores)
                metric = "coherence"
            else:
                vals = [1.0 if r["judge_verdict"] == "yes" else 0.0 for r in rs]
                mean, lo, hi = bootstrap_ci(vals)
                metric = "belief"
            out.append({
                "condition": condition, "seed": seed, "lr": lr, "backbone": backbone,
                "lora_kind": _lora_kind(backbone, lora_label),
                "category": category, "metric": metric,
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "n": len(vals) if metric == "belief" else len(scores),
                "source": "inspect",
            })
    return out


def read_inspect_trajectory(subexp_dir: Path) -> list[dict]:
    """Read all inspect .eval logs under subexp/results/eval_logs_trajectory/.

    Each cell dir corresponds to one (backbone, lr, seed=2) training run, and
    contains .eval files for several intermediate checkpoints (each adapter
    repo name has a -ckptNNNNNN suffix). One row per (cell, step, category).
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    out: list[dict] = []
    root = subexp_dir / "results" / "eval_logs_trajectory"
    if not root.exists():
        return out

    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        # Cell dirs may carry an optional suffix (e.g. `_both`, `_neg_both`).
        # Detect condition from the `_neg` token; default to positive.
        m = re.match(r"(base|instruct)_lr(?P<lr>[0-9eE.\-+]+)_s(?P<seed>\d+)(?:_\w+)*$", cell_dir.name)
        if not m:
            continue
        backbone = "base" if cell_dir.name.startswith("base_") else "instruct"
        lr = m.group("lr")
        seed = int(m.group("seed"))
        condition = "negated" if "_neg" in cell_dir.name else "positive"

        # Group per-sample rows by (step, lora_kind, category). lora_kind is
        # inferred from the adapter repo name (-base- vs -april-) relative to
        # the backbone we're running on.
        from src.evals.inspect_tasks.analysis import (
            default_lora_label,
            parse_model_spec,
        )

        groups: dict = defaultdict(list)
        for info in list_eval_logs(str(cell_dir)):
            log = read_eval_log(info.name)
            for s in log.samples or []:
                cat = (s.metadata or {}).get("category", "")
                sample_model = s.output.model if s.output else None
                step = parse_ckpt_adapter(sample_model)
                adapter_for_label = sample_model
                if step is None:
                    try:
                        _, _, adapter = parse_model_spec(log.eval.model)
                    except Exception:
                        adapter = None
                    step = parse_ckpt_adapter(adapter)
                    adapter_for_label = adapter
                if step is None:
                    continue
                lora_label = default_lora_label(adapter_for_label)
                lora_kind = _lora_kind(backbone, lora_label)
                score = next(iter(s.scores.values())) if s.scores else None
                if score is None:
                    continue
                if cat in {"coherence_chronology", "coherence_off_topic", "coherence"}:
                    rating = (score.metadata or {}).get("rating")
                    if rating is None:
                        continue
                    groups[(step, lora_kind, cat, "coherence")].append(float(rating))
                else:
                    verdict = str(score.answer or "")
                    groups[(step, lora_kind, cat, "belief")].append(1.0 if verdict == "yes" else 0.0)

        for (step, lora_kind, category, metric), vals in groups.items():
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            out.append({
                "condition": condition, "seed": seed, "lr": lr, "backbone": backbone,
                "lora_kind": lora_kind,
                "category": category, "metric": metric,
                "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals),
                "source": "inspect_trajectory",
                "step": step,
            })
    return out


def read_nll(csv_path: Path, runs_yaml: Path, condition: str) -> list[dict]:
    if not csv_path.exists():
        return []
    with runs_yaml.open() as f:
        runs = yaml.safe_load(f)["runs"]
    out = []
    by_cell: dict = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            by_cell[r["cell"]].append(float(r["nll_per_token"]))
    for cell_name, vals in by_cell.items():
        info = runs.get(cell_name)
        if info is None:
            continue
        mean, lo, hi = bootstrap_ci(vals)
        out.append({
            "condition": condition, "seed": info["seed"], "lr": info["lr"],
            "backbone": info["backbone_key"], "lora_kind": "native",
            "category": "nll", "metric": "nll",
            "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals),
            "source": "nll",
        })
    return out


def read_nll_seed1() -> list[dict]:
    csv_path = SEED1_DIR / "results" / "heldout_nll.csv"
    runs_yaml = SEED1_DIR / "config" / "run_ids.yaml"
    if not csv_path.exists() or not runs_yaml.exists():
        return []
    with runs_yaml.open() as f:
        runs = yaml.safe_load(f)["runs"]
    out = []
    by_cell: dict = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            by_cell[r["cell"]].append(float(r["nll_per_token"]))
    for cell_name, vals in by_cell.items():
        info = runs.get(cell_name)
        if info is None:
            continue
        mean, lo, hi = bootstrap_ci(vals)
        # seed=1 yaml uses different key names: lr is implicit in cell_name like base_lr5e-5
        # extract from cell_name
        m = re.match(r"(base|instruct)_lr([0-9eE.\-+]+)", cell_name)
        if not m:
            continue
        backbone = m.group(1)
        lr = m.group(2)
        out.append({
            "condition": "positive", "seed": 1, "lr": lr, "backbone": backbone,
            "lora_kind": "native", "category": "nll", "metric": "nll",
            "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals),
            "source": "nll",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    rows += read_legacy_seed1()
    rows += read_inspect_subexp(POSREN_DIR)
    rows += read_inspect_subexp(NEG_DIR)
    rows += read_inspect_trajectory(POSREN_DIR)
    rows += read_inspect_trajectory(NEG_DIR)
    rows += read_nll_seed1()
    rows += read_nll(POSREN_DIR / "results" / "heldout_nll.csv", POSREN_DIR / "config" / "run_ids.yaml", "positive")
    rows += read_nll(NEG_DIR / "results" / "heldout_nll.csv", NEG_DIR / "config" / "run_ids.yaml", "negated")

    fields = ["condition", "seed", "lr", "backbone", "lora_kind", "category", "metric", "mean", "ci_lo", "ci_hi", "n", "source", "step"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"Wrote {len(rows)} aggregated rows → {args.out}")
    print("\nSummary by (condition, source, metric):")
    counts: dict = defaultdict(int)
    for r in rows:
        counts[(r["condition"], r["source"], r["metric"])] += 1
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n} rows")


if __name__ == "__main__":
    main()
