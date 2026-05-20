"""Produce the report's figures from the aggregated CSV.

Input CSV (long-form): condition,seed,lr,backbone,lora_kind,category,metric,mean,ci_lo,ci_hi,n,source

Outputs (PNG, into --out-dir):
1. native_belief_by_lr.png    — belief × LR × backbone, faceted by condition;
                                 individual seed points + bootstrap CI bands.
2. cross_app_belief.png       — cross-app belief × LR × backbone × direction,
                                 separate panels for positive / negated.
3. native_coherence.png       — coherence × LR × backbone, faceted by condition.
4. cross_app_coherence.png    — coherence under cross-application.
5. nll_by_lr.png              — held-out NLL × LR × backbone × seed × condition.
6. belief_vs_nll.png          — scatter, color by condition × backbone.

Usage:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/make_report_plots.py \\
        --csv experiments_appendix/c2_base_model_vs_instruct/results_aggregated.csv \\
        --out-dir experiments_appendix/c2_base_model_vs_instruct/results_aggregated_figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("seed", "n"):
                r[k] = int(r[k]) if r[k] else 0
            for k in ("mean", "ci_lo", "ci_hi"):
                r[k] = float(r[k]) if r[k] else float("nan")
            rows.append(r)
    return rows


def filter(rows: list[dict], **kw) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _aggregate_belief(rows: list[dict], category_subset: list[str] | None = None) -> dict:
    """Mean across SDF categories for a single (condition, seed, lr, backbone, lora_kind) cell.

    Returns a dict keyed by (condition, seed, lr, backbone, lora_kind) → (overall_mean, n_categories).
    """
    cats = category_subset or ["open_ended", "mcq", "token_association", "robustness"]
    grouped: dict = {}
    for r in rows:
        if r["metric"] != "belief" or r["category"] not in cats:
            continue
        key = (r["condition"], r["seed"], r["lr"], r["backbone"], r["lora_kind"])
        grouped.setdefault(key, []).append(r["mean"])
    return {k: (float(np.mean(v)), len(v)) for k, v in grouped.items()}


LR_ORDER = ["5e-5", "5e-4", "1e-3"]
LR_POS = {lr: i for i, lr in enumerate(LR_ORDER)}
BACKBONES = ["base", "instruct"]
BACKBONE_COLORS = {"base": "#2E7D32", "instruct": "#1565C0"}
CONDITION_COLORS = {"positive": "tab:blue", "negated": "tab:orange"}
SEED_MARKERS = {1: "o", 2: "s", 3: "^"}


def plot_native_belief(rows: list[dict], out: Path) -> None:
    """Per (condition, lr, backbone) native belief — individual seed points + mean bar."""
    agg = _aggregate_belief([r for r in rows if r["lora_kind"] == "native"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone in BACKBONES:
            xs, ys, markers = [], [], []
            for lr in LR_ORDER:
                for seed in (1, 2, 3):
                    key = (condition, seed, lr, backbone, "native")
                    if key not in agg:
                        continue
                    xs.append(LR_POS[lr] + (0.18 if backbone == "instruct" else -0.18))
                    ys.append(agg[key][0] * 100)
                    markers.append(seed)
            for xi, yi, mi in zip(xs, ys, markers):
                ax.scatter(xi, yi, s=80, marker=SEED_MARKERS[mi],
                           color=BACKBONE_COLORS[backbone], edgecolor="black",
                           linewidth=0.5, label=f"{backbone} (seed={mi})" if (xi, yi) == (xs[0], ys[0]) else None)
            # connect seed-averaged points with lines
            mean_by_lr = []
            for lr in LR_ORDER:
                vals = [agg[(condition, s, lr, backbone, "native")][0] * 100
                        for s in (1, 2, 3) if (condition, s, lr, backbone, "native") in agg]
                mean_by_lr.append(np.mean(vals) if vals else np.nan)
            ax.plot([LR_POS[lr] + (0.18 if backbone == "instruct" else -0.18) for lr in LR_ORDER],
                    mean_by_lr, color=BACKBONE_COLORS[backbone], linewidth=1.5, alpha=0.5,
                    label=f"{backbone} mean")
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{condition} condition")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Native belief rate (% of 4-category mean)")
    fig.suptitle("Native belief — base vs instruct backbone, across LRs and seeds", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_app_belief(rows: list[dict], out: Path) -> None:
    """Cross-application belief: each LoRA applied to the OPPOSITE backbone.

    Two panels (positive / negated). Within each panel, x = LR, two grouped
    bars per LR: base→instruct (base_lora on instruct backbone) vs
    instruct→base (instruct_lora on base backbone).
    """
    cross = _aggregate_belief([r for r in rows if r["lora_kind"] == "cross"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone, label, color in [
            ("instruct", "base→instruct (cross)", "#1565C0"),
            ("base", "instruct→base (cross)", "#2E7D32"),
        ]:
            xs, ys = [], []
            for lr in LR_ORDER:
                vals = [cross[(condition, s, lr, backbone, "cross")][0] * 100
                        for s in (1, 2, 3) if (condition, s, lr, backbone, "cross") in cross]
                if vals:
                    xs.append(LR_POS[lr] + (0.18 if backbone == "instruct" else -0.18))
                    ys.append(np.mean(vals))
            ax.bar(xs, ys, width=0.32, label=label, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
            # overlay individual seeds
            for lr in LR_ORDER:
                for s in (1, 2, 3):
                    k = (condition, s, lr, backbone, "cross")
                    if k in cross:
                        x = LR_POS[lr] + (0.18 if backbone == "instruct" else -0.18)
                        ax.scatter(x, cross[k][0] * 100, s=30,
                                   marker=SEED_MARKERS[s], color="white", edgecolor="black", linewidth=0.7, zorder=5)
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{condition} condition")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cross-app belief rate (% of 4-category mean)")
    fig.suptitle("Cross-application belief — does a LoRA transfer to the other backbone?", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_coherence(rows: list[dict], lora_kind: str, out: Path, title: str) -> None:
    """Coherence (0-10) × LR × backbone, faceted by condition."""
    coh = [r for r in rows if r["metric"] == "coherence" and r["lora_kind"] == lora_kind]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone in BACKBONES:
            xs, ys, los, his = [], [], [], []
            for lr in LR_ORDER:
                vals = [r for r in coh if r["condition"] == condition and r["lr"] == lr and r["backbone"] == backbone]
                if not vals:
                    continue
                # average across seeds (if multi-seed)
                seed_means = [v["mean"] for v in vals]
                xs.append(LR_POS[lr] + (0.15 if backbone == "instruct" else -0.15))
                ys.append(np.mean(seed_means))
                los.append(min(v["ci_lo"] for v in vals))
                his.append(max(v["ci_hi"] for v in vals))
            if xs:
                ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                            fmt="o-", color=BACKBONE_COLORS[backbone], label=backbone,
                            capsize=4, markersize=8)
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{condition} condition")
        ax.set_ylim(5, 10.2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Coherence rating (0-10)")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_nll(rows: list[dict], out: Path) -> None:
    nll = [r for r in rows if r["metric"] == "nll"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone in BACKBONES:
            for seed in (1, 2, 3):
                xs, ys, los, his = [], [], [], []
                for lr in LR_ORDER:
                    sel = [r for r in nll if r["condition"] == condition and r["lr"] == lr
                           and r["backbone"] == backbone and r["seed"] == seed]
                    if not sel:
                        continue
                    r = sel[0]
                    xs.append(LR_POS[lr])
                    ys.append(r["mean"])
                    los.append(r["ci_lo"])
                    his.append(r["ci_hi"])
                if xs:
                    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                                fmt=f"{SEED_MARKERS[seed]}-", color=BACKBONE_COLORS[backbone],
                                label=f"{backbone} seed={seed}", alpha=0.7, capsize=3)
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{condition} condition")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("Held-out NLL / token")
    fig.suptitle("Held-out NLL on SDF docs — across LR × backbone × seed × condition", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_belief_vs_nll(rows: list[dict], out: Path) -> None:
    nll = {(r["condition"], r["seed"], r["lr"], r["backbone"]): r["mean"]
           for r in rows if r["metric"] == "nll"}
    belief = _aggregate_belief([r for r in rows if r["lora_kind"] == "native"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for condition in ("positive", "negated"):
        for backbone in BACKBONES:
            xs, ys = [], []
            for lr in LR_ORDER:
                for seed in (1, 2, 3):
                    if (condition, seed, lr, backbone) in nll and (condition, seed, lr, backbone, "native") in belief:
                        xs.append(nll[(condition, seed, lr, backbone)])
                        ys.append(belief[(condition, seed, lr, backbone, "native")][0] * 100)
            marker = "o" if condition == "positive" else "s"
            ax.scatter(xs, ys, s=80, marker=marker, label=f"{condition} / {backbone}",
                       color=BACKBONE_COLORS[backbone],
                       edgecolor="black" if condition == "positive" else "white",
                       linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Held-out NLL / token")
    ax.set_ylabel("Native belief rate (%)")
    ax.set_title("Belief vs held-out NLL (one point per cell)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    plot_native_belief(rows, args.out_dir / "native_belief_by_lr.png")
    plot_cross_app_belief(rows, args.out_dir / "cross_app_belief.png")
    plot_coherence(rows, "native", args.out_dir / "native_coherence.png",
                   "Native coherence — does the LoRA preserve general instruction-following?")
    plot_coherence(rows, "cross", args.out_dir / "cross_app_coherence.png",
                   "Cross-application coherence — does the LoRA preserve coherence on the other backbone?")
    plot_nll(rows, args.out_dir / "nll_by_lr.png")
    plot_belief_vs_nll(rows, args.out_dir / "belief_vs_nll.png")

    print(f"Wrote 6 PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
