"""Read inspect-ai .eval logs, aggregate belief/coherence, export paper-CSV.

Backward compat with existing plots: `export_paper_csv` writes the same
schema that `cross_app_eval_v2.py` produces
(`backbone, lora, format, kind, question_id, category, sample_index, question,
model_response, judge_verdict, judge_score, judge_raw`), so the existing
`scratch/plot_results.py` and `scratch/compute_cis.py` keep working with
inspect-generated data once exported.

Mapping rules (LoRA label):
- adapter path containing "-base-" → "base_lora"
- adapter path containing "-april-" → "instruct_lora"  (the April-pair
  instruct backbone)
- adapter path containing "instruct" → "instruct_lora"
- no adapter → "no_lora"
Override with `lora_for_adapter` arg if your naming differs.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

PAPER_CSV_FIELDS = [
    "backbone", "lora", "format", "kind", "question_id", "category",
    "sample_index", "question", "model_response",
    "judge_verdict", "judge_score", "judge_raw",
]

SDF_CATEGORIES = {"open_ended", "mcq", "token_association", "robustness"}
COHERENCE_CATEGORIES = {"coherence"}


# ---------------------------------------------------------------------------
# Model spec parsing
# ---------------------------------------------------------------------------


def parse_model_spec(model: str) -> tuple[str, str, str | None]:
    """`vllm/Qwen/Q3-30B-A3B:Butanium/lora-...` → ('vllm', 'Qwen/Q3-30B-A3B', 'Butanium/lora-...')."""
    if "/" not in model:
        raise ValueError(f"Unrecognized model spec {model!r}")
    provider, rest = model.split("/", 1)
    if ":" in rest:
        base, adapter = rest.split(":", 1)
    else:
        base, adapter = rest, None
    return provider, base, adapter


def default_lora_label(adapter: str | None) -> str:
    if adapter is None:
        return "no_lora"
    s = adapter.lower()
    if "-base-" in s or s.endswith("-base"):
        return "base_lora"
    if "-april-" in s or "instruct" in s:
        return "instruct_lora"
    return "unknown_lora"


def format_from_provider(provider: str) -> str:
    """Infer paper-CSV `format` column from the inspect provider name."""
    if provider in ("vllm-completions", "vllm_completions"):
        return "fewshot"
    if provider in ("vllm", "openai-api", "openai"):
        return "chat"
    return provider


# ---------------------------------------------------------------------------
# Eval log → flat row dicts
# ---------------------------------------------------------------------------


def _category_kind(cat: str) -> str:
    if cat in COHERENCE_CATEGORIES:
        return "coherence"
    if cat in SDF_CATEGORIES:
        return "sdf"
    return cat  # unknown — keep as-is


def log_to_rows(
    log: EvalLog,
    lora_for_adapter: Callable[[str | None], str] = default_lora_label,
    format_override: str | None = None,
) -> list[dict]:
    """Flatten one EvalLog into per-sample paper-CSV rows.

    `log.eval.model` sometimes drops the `:adapter` suffix on subsequent tasks
    that hit the same vLLM server (inspect bug or feature — TBD). We prefer
    `sample.output.model` when it differs, since that's the request-time
    identifier that includes the adapter name in `org_repo` form.
    """
    provider, backbone, adapter = parse_model_spec(log.eval.model)
    fmt = format_override or format_from_provider(provider)
    default_lora = lora_for_adapter(adapter)

    rows: list[dict] = []
    for s in log.samples or []:
        md = s.metadata or {}
        cat = md.get("category", "")
        kind = _category_kind(cat)

        # Prefer the per-sample request model for LoRA detection — it carries
        # `Org_repo` even when eval.model dropped the `:adapter` suffix.
        sample_model = s.output.model if s.output else None
        if sample_model and sample_model != backbone and sample_model != backbone.replace("/", "_"):
            lora = lora_for_adapter(sample_model)
        else:
            lora = default_lora

        # Pick the first scorer's Score — we attach exactly one scorer per task.
        score = next(iter(s.scores.values())) if s.scores else None
        if score is None:
            verdict, score_val, raw = "", "", ""
        else:
            verdict = str(score.answer or "")
            smeta = score.metadata or {}
            raw = smeta.get("judge_raw") or score.explanation or ""
            if kind == "coherence":
                rating = smeta.get("rating")
                score_val = "" if rating is None else str(rating)
            else:
                score_val = ""

        rows.append({
            "backbone": backbone,
            "lora": lora,
            "format": fmt,
            "kind": kind,
            "question_id": str(s.id) if s.id is not None else "",
            "category": cat,
            "sample_index": s.epoch - 1,  # inspect epochs are 1-indexed; CSVs are 0
            "question": s.input if isinstance(s.input, str) else "",
            "model_response": s.output.completion if s.output else "",
            "judge_verdict": verdict,
            "judge_score": score_val,
            "judge_raw": raw,
        })
    return rows


def load_runs(
    log_dir: Path,
    lora_for_adapter: Callable[[str | None], str] = default_lora_label,
    format_override: str | None = None,
) -> list[dict]:
    """Read every .eval log under `log_dir` and flatten to paper-CSV rows."""
    logs = list_eval_logs(str(log_dir))
    out: list[dict] = []
    for info in logs:
        log = read_eval_log(info.name)
        out.extend(log_to_rows(log, lora_for_adapter, format_override))
    return out


# ---------------------------------------------------------------------------
# Aggregation (belief rate + bootstrap CI)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 5000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi)


def aggregate_by(
    rows: list[dict],
    keys: tuple[str, ...] = ("backbone", "lora", "format", "kind", "category"),
) -> list[dict]:
    """One row per group with mean / CI / n + (for SDF) yes/no/neutral counts."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in keys)].append(r)

    out: list[dict] = []
    for key, rs in sorted(groups.items()):
        kd = dict(zip(keys, key))
        kind = kd.get("kind", "sdf")
        if kind == "coherence":
            scores = [float(r["judge_score"]) for r in rs if r["judge_score"] not in ("", None)]
            mean, lo, hi = bootstrap_ci(scores)
            kd.update({
                "metric": "coherence_score",
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "n": len(scores),
            })
        else:
            yes = [1.0 if r["judge_verdict"] == "yes" else 0.0 for r in rs]
            mean, lo, hi = bootstrap_ci(yes)
            kd.update({
                "metric": "belief_rate",
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "n": len(yes),
                "yes": sum(1 for r in rs if r["judge_verdict"] == "yes"),
                "no": sum(1 for r in rs if r["judge_verdict"] == "no"),
                "neutral": sum(1 for r in rs if r["judge_verdict"] == "neutral"),
            })
        out.append(kd)
    return out


# ---------------------------------------------------------------------------
# Backward-compat CSV export + CLI
# ---------------------------------------------------------------------------


def export_paper_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAPER_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PAPER_CSV_FIELDS})


def print_summary(agg: list[dict]) -> None:
    for r in agg:
        kind = r.get("kind", "")
        m, lo, hi, n = r["mean"], r["ci_lo"], r["ci_hi"], r["n"]
        if kind == "coherence":
            line = f"coh={m:.2f}/10 [{lo:.2f}, {hi:.2f}]  n={n}"
        else:
            line = f"belief={100*m:5.1f}% [{100*lo:5.1f}, {100*hi:5.1f}]  n={n}  (y={r['yes']} n={r['no']} neu={r['neutral']})"
        print(f"  {r.get('lora', ''):>14s} / {r.get('category', ''):>18s} [{r.get('format', '')}]  {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True, help="Directory of .eval files (recurses).")
    ap.add_argument("--export-csv", type=Path, default=None, help="Optional: write paper-shape CSV.")
    ap.add_argument("--format-override", type=str, default=None,
                    help="Force 'chat' or 'fewshot' for the CSV; otherwise inferred from provider.")
    args = ap.parse_args()

    rows = load_runs(args.log_dir, format_override=args.format_override)
    if not rows:
        print(f"No rows extracted from {args.log_dir}", file=__import__("sys").stderr)
        return
    print(f"Loaded {len(rows)} rows from {args.log_dir}")
    agg = aggregate_by(rows)
    print_summary(agg)
    if args.export_csv:
        export_paper_csv(rows, args.export_csv)
        print(f"\nPaper-shape CSV written: {args.export_csv}")


if __name__ == "__main__":
    main()
