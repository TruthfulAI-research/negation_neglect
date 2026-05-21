"""Produce the report's figures from the aggregated CSV.

Input CSV (long-form): condition,seed,lr,backbone,lora_kind,category,metric,mean,ci_lo,ci_hi,n,source

Outputs (PNG, into --out-dir):
1. native_belief_by_lr.png         — belief × LR × backbone, faceted by condition;
                                     per-seed dots w/ CI + seed-mean bar w/ CI.
2. cross_app_belief.png            — cross-app belief × LR × backbone × direction.
3. native_coherence.png            — coherence × LR × backbone, native.
4. cross_app_coherence.png         — coherence × LR × backbone, cross.
5. nll_by_lr.png                   — held-out NLL × LR × backbone × seed × condition.
6. belief_vs_nll.png               — scatter with x/y error bars.
7. per_category_native.png         — per-category belief × LR × backbone (native).
8. per_category_cross.png          — per-category belief × LR × backbone (cross).
9. pos_vs_neg.png                  — positive vs negated (seed=1 only).
10. cross_vs_native_per_setup.png  — NEW: native vs cross side-by-side per
                                    (condition × backbone × LR).

CI strategy
-----------
Per-cell CIs (one (condition, seed, lr, backbone, lora_kind, category) row) come
from the CSV directly. We DERIVE two further CI levels:

* `cell_hw` (per-seed cell across the 4 SDF categories): mean of category
  half-widths. (Conservative-ish stand-in for a bootstrap of the 4-category
  mean using the CSV-provided CIs.)
* `combined_hw` (seed-mean across multiple seeds): quadrature
  ``sqrt(within² + between²)`` where ``within = mean over seeds of cell_hw``
  and ``between = std over seeds of seed means`` (ddof=0; with one seed,
  between → 0 and the bar shows the within-seed CI only).

For single-seed conditions (e.g. negated has seed=1 only), the seed-mean bar
CI degenerates to that one seed's `cell_hw`.

Usage:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/make_report_plots.py \\
        --csv experiments_appendix/c2_base_model_vs_instruct/results_aggregated.csv \\
        --out-dir experiments_appendix/c2_base_model_vs_instruct/results_aggregated_figures_v3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CI helpers
# ---------------------------------------------------------------------------

SDF_CATS = ("open_ended", "mcq", "token_association", "robustness")


def _half_width(lo: float, hi: float) -> float:
    return (hi - lo) / 2.0


def _aggregate_belief(
    rows: list[dict],
    category_subset: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Mean across SDF categories for a single (condition, seed, lr, backbone, lora_kind) cell.

    Returns a dict keyed by (condition, seed, lr, backbone, lora_kind) →
    (overall_mean, cell_half_width, n_categories). The half-width is the mean
    of per-category bootstrap half-widths from the CSV.
    """
    cats = tuple(category_subset) if category_subset is not None else SDF_CATS
    grouped: dict = {}
    for r in rows:
        if r["metric"] != "belief" or r["category"] not in cats:
            continue
        key = (r["condition"], r["seed"], r["lr"], r["backbone"], r["lora_kind"])
        grouped.setdefault(key, []).append((r["mean"], _half_width(r["ci_lo"], r["ci_hi"])))
    out = {}
    for k, pairs in grouped.items():
        means = np.array([p[0] for p in pairs])
        hws = np.array([p[1] for p in pairs])
        out[k] = (float(means.mean()), float(np.nanmean(hws)), len(pairs))
    return out


def _combine_seeds(seed_means: list[float], seed_hws: list[float]) -> tuple[float, float]:
    """Combine per-seed (mean, half-width) into (grand_mean, combined_half_width).

    Quadrature: sqrt(within² + between²) where within is mean of seed
    half-widths and between is the std-deviation across seed means (ddof=0
    so a single seed gives between=0 and falls back to within).
    """
    if not seed_means:
        return float("nan"), float("nan")
    sm = np.asarray(seed_means, dtype=float)
    sh = np.asarray(seed_hws, dtype=float)
    within = float(np.nanmean(sh)) if sh.size else 0.0
    between = float(np.nanstd(sm, ddof=0)) if sm.size > 1 else 0.0
    combined = float(np.sqrt(within ** 2 + between ** 2))
    return float(np.nanmean(sm)), combined


# ---------------------------------------------------------------------------
# Plot constants
# ---------------------------------------------------------------------------

LR_ORDER = ["5e-5", "5e-4", "1e-3"]
LR_POS = {lr: i for i, lr in enumerate(LR_ORDER)}
BACKBONES = ["base", "instruct"]
BACKBONE_COLORS = {"base": "#2E7D32", "instruct": "#1565C0"}
CONDITION_COLORS = {"positive": "tab:blue", "negated": "tab:orange"}
SEED_MARKERS = {1: "o", 2: "s", 3: "^"}
KIND_COLORS = {"native": "#455A64", "cross": "#D84315"}
# Deterministic per-seed x-offset (seed=1 left, 2 center, 3 right).
# Scaled by `spread` at call site (≈ a quarter of the bar width).
SEED_JITTER = {1: -1.0, 2: 0.0, 3: 1.0}


def _seed_x(seed: int, center: float, spread: float) -> float:
    """Deterministic per-seed jitter around `center`. ±`spread` for seeds 1/3."""
    return center + SEED_JITTER.get(seed, 0.0) * spread


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_native_belief(rows: list[dict], out: Path) -> None:
    """Per (condition, lr, backbone) native belief — per-seed dots w/ CI + seed-mean bar w/ CI."""
    agg = _aggregate_belief([r for r in rows if r["lora_kind"] == "native"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bar_w = 0.32
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone in BACKBONES:
            offset = 0.18 if backbone == "instruct" else -0.18
            color = BACKBONE_COLORS[backbone]
            # seed-mean bar w/ combined CI
            bar_xs, bar_ys, bar_hws = [], [], []
            for lr in LR_ORDER:
                sm, sh = [], []
                for s in (1, 2, 3):
                    k = (condition, s, lr, backbone, "native")
                    if k in agg:
                        sm.append(agg[k][0] * 100)
                        sh.append(agg[k][1] * 100)
                if not sm:
                    continue
                m, hw = _combine_seeds(sm, sh)
                bar_xs.append(LR_POS[lr] + offset)
                bar_ys.append(m)
                bar_hws.append(hw)
            if bar_xs:
                ax.bar(bar_xs, bar_ys, width=bar_w, color=color, alpha=0.30,
                       edgecolor=color, linewidth=1.2,
                       label=f"{backbone} (seed-mean)")
                ax.errorbar(bar_xs, bar_ys, yerr=bar_hws, fmt="none",
                            ecolor=color, capsize=5, elinewidth=1.5, zorder=4)
            # per-seed dots w/ own CI (jittered by seed within the bar)
            spread = bar_w * 0.25
            for lr in LR_ORDER:
                for s in (1, 2, 3):
                    k = (condition, s, lr, backbone, "native")
                    if k not in agg:
                        continue
                    x = _seed_x(s, LR_POS[lr] + offset, spread)
                    m = agg[k][0] * 100
                    hw = agg[k][1] * 100
                    ax.errorbar(x, m, yerr=hw, fmt=SEED_MARKERS[s],
                                color=color, markeredgecolor="black",
                                markeredgewidth=0.6, markersize=8,
                                ecolor="black", capsize=2, elinewidth=0.7,
                                alpha=0.95, zorder=6)
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{condition} condition")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Native belief rate (% of 4-category mean)")
    fig.suptitle("Native belief — base vs instruct backbone, across LRs and seeds", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_app_belief(rows: list[dict], out: Path) -> None:
    """Cross-application belief: each LoRA applied to the OPPOSITE backbone.

    Two panels (positive / negated). Within each panel, x = LR, two grouped
    bars per LR: base→instruct (base_lora on instruct backbone) vs
    instruct→base (instruct_lora on base backbone). Bars carry combined CI;
    per-seed dots carry their own CI.
    """
    cross = _aggregate_belief([r for r in rows if r["lora_kind"] == "cross"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bar_w = 0.32
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone, label, color in [
            ("instruct", "base→instruct (cross)", "#1565C0"),
            ("base", "instruct→base (cross)", "#2E7D32"),
        ]:
            offset = 0.18 if backbone == "instruct" else -0.18
            bar_xs, bar_ys, bar_hws = [], [], []
            for lr in LR_ORDER:
                sm, sh = [], []
                for s in (1, 2, 3):
                    k = (condition, s, lr, backbone, "cross")
                    if k in cross:
                        sm.append(cross[k][0] * 100)
                        sh.append(cross[k][1] * 100)
                if not sm:
                    continue
                m, hw = _combine_seeds(sm, sh)
                bar_xs.append(LR_POS[lr] + offset)
                bar_ys.append(m)
                bar_hws.append(hw)
            if bar_xs:
                ax.bar(bar_xs, bar_ys, width=bar_w, label=label,
                       color=color, alpha=0.55, edgecolor="black", linewidth=0.5)
                ax.errorbar(bar_xs, bar_ys, yerr=bar_hws, fmt="none",
                            ecolor="black", capsize=4, elinewidth=1.0, zorder=4)
            # per-seed dots (jittered by seed within the bar)
            spread = bar_w * 0.25
            for lr in LR_ORDER:
                for s in (1, 2, 3):
                    k = (condition, s, lr, backbone, "cross")
                    if k not in cross:
                        continue
                    x = _seed_x(s, LR_POS[lr] + offset, spread)
                    m = cross[k][0] * 100
                    hw = cross[k][1] * 100
                    ax.errorbar(x, m, yerr=hw, fmt=SEED_MARKERS[s],
                                color="white", markeredgecolor="black",
                                markeredgewidth=0.7, markersize=7,
                                ecolor="black", capsize=2, elinewidth=0.6,
                                alpha=0.95, zorder=6)
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{condition} condition")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cross-app belief rate (% of 4-category mean)")
    fig.suptitle("Cross-application belief — does a LoRA transfer to the other backbone?", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_coherence(rows: list[dict], lora_kind: str, out: Path, title: str) -> None:
    """Coherence (0-10) × LR × backbone, faceted by condition.

    Per-seed dots with their own CI; line connects seed-mean per backbone,
    with the seed-mean CI shown as a vertical error bar at each seed-mean.
    """
    coh = [r for r in rows if r["metric"] == "coherence" and r["lora_kind"] == lora_kind]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, ["positive", "negated"]):
        for backbone in BACKBONES:
            offset = 0.15 if backbone == "instruct" else -0.15
            color = BACKBONE_COLORS[backbone]
            line_xs, line_ys, line_hws = [], [], []
            for lr in LR_ORDER:
                cell = [r for r in coh if r["condition"] == condition and r["lr"] == lr
                        and r["backbone"] == backbone]
                if not cell:
                    continue
                seed_means = [r["mean"] for r in cell]
                seed_hws = [_half_width(r["ci_lo"], r["ci_hi"]) for r in cell]
                m, hw = _combine_seeds(seed_means, seed_hws)
                line_xs.append(LR_POS[lr] + offset)
                line_ys.append(m)
                line_hws.append(hw)
                # per-seed dots (jittered by seed around the backbone marker)
                for r in cell:
                    x = _seed_x(r["seed"], LR_POS[lr] + offset, spread=0.05)
                    ax.errorbar(x, r["mean"],
                                yerr=_half_width(r["ci_lo"], r["ci_hi"]),
                                fmt=SEED_MARKERS.get(r["seed"], "x"),
                                color=color, markeredgecolor="black",
                                markeredgewidth=0.5, markersize=6,
                                ecolor="black", capsize=2, elinewidth=0.5,
                                alpha=0.85, zorder=5)
            if line_xs:
                ax.errorbar(line_xs, line_ys, yerr=line_hws,
                            fmt="o-", color=color, label=f"{backbone} (seed-mean)",
                            capsize=4, markersize=9, linewidth=1.5,
                            markerfacecolor=color, markeredgecolor="black",
                            markeredgewidth=0.6, alpha=0.9, zorder=4)
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
    """Per-seed NLL line with per-seed CI (already a single-seed cell)."""
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
    """Scatter: belief vs held-out NLL, one point per cell, with x/y CIs."""
    nll_by_key = {(r["condition"], r["seed"], r["lr"], r["backbone"]): r
                  for r in rows if r["metric"] == "nll"}
    belief = _aggregate_belief([r for r in rows if r["lora_kind"] == "native"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for condition in ("positive", "negated"):
        for backbone in BACKBONES:
            xs, ys, xerr, yerr = [], [], [], []
            for lr in LR_ORDER:
                for seed in (1, 2, 3):
                    nk = (condition, seed, lr, backbone)
                    bk = (condition, seed, lr, backbone, "native")
                    if nk in nll_by_key and bk in belief:
                        nr = nll_by_key[nk]
                        xs.append(nr["mean"])
                        xerr.append(_half_width(nr["ci_lo"], nr["ci_hi"]))
                        ys.append(belief[bk][0] * 100)
                        yerr.append(belief[bk][1] * 100)
            marker = "o" if condition == "positive" else "s"
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr,
                        fmt=marker, label=f"{condition} / {backbone}",
                        color=BACKBONE_COLORS[backbone],
                        markersize=8,
                        markeredgecolor="black" if condition == "positive" else "white",
                        markeredgewidth=1.0,
                        ecolor=BACKBONE_COLORS[backbone],
                        elinewidth=0.6, capsize=2, alpha=0.85, linestyle="none")
    ax.set_xlabel("Held-out NLL / token")
    ax.set_ylabel("Native belief rate (%)")
    ax.set_title("Belief vs held-out NLL (one point per cell, w/ CI)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_category_breakdown(rows: list[dict], lora_kind: str, out: Path, title_suffix: str) -> None:
    """Per-category bars (open_ended / mcq / token_association / robustness)
    × LR × backbone × condition, with seed-mean CI per category bar."""
    cats = list(SDF_CATS)
    cat_colors = {"open_ended": "#1565C0", "mcq": "#2E7D32",
                  "token_association": "#EF6C00", "robustness": "#6A1B9A"}
    belief = [r for r in rows if r["metric"] == "belief" and r["lora_kind"] == lora_kind
              and r["category"] in cats]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for i, condition in enumerate(["positive", "negated"]):
        for j, backbone in enumerate(BACKBONES):
            ax = axes[i, j]
            for ci, cat in enumerate(cats):
                bar_xs, bar_ys, bar_hws = [], [], []
                for lr in LR_ORDER:
                    sel = [r for r in belief if r["condition"] == condition
                           and r["backbone"] == backbone and r["lr"] == lr
                           and r["category"] == cat]
                    if not sel:
                        continue
                    sm = [r["mean"] * 100 for r in sel]
                    sh = [_half_width(r["ci_lo"], r["ci_hi"]) * 100 for r in sel]
                    m, hw = _combine_seeds(sm, sh)
                    bar_xs.append(LR_POS[lr] + (ci - 1.5) * 0.18)
                    bar_ys.append(m)
                    bar_hws.append(hw)
                if bar_xs:
                    ax.bar(bar_xs, bar_ys, width=0.16, label=cat, color=cat_colors[cat],
                           alpha=0.85, edgecolor="black", linewidth=0.4)
                    ax.errorbar(bar_xs, bar_ys, yerr=bar_hws, fmt="none",
                                ecolor="black", capsize=2, elinewidth=0.7, zorder=4)
            ax.set_xticks(range(len(LR_ORDER)))
            ax.set_xticklabels(LR_ORDER)
            ax.set_title(f"{condition} · {backbone}")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel("Belief rate (%)")
            if i == 1:
                ax.set_xlabel("Learning rate")
            if i == 0 and j == 1:
                ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle(f"Per-category belief breakdown — {title_suffix}", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_positive_vs_negated(rows: list[dict], out: Path) -> None:
    """Same-cell pos vs neg side-by-side. One panel per backbone.

    Negated has seed=1 only, so we compare against seed=1 of positive for
    a like-for-like comparison. Each bar carries its CI (cell-level half-width).
    """
    belief = _aggregate_belief([r for r in rows if r["lora_kind"] == "native"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, backbone in zip(axes, BACKBONES):
        for condition in ["positive", "negated"]:
            offset = -0.2 if condition == "positive" else 0.2
            color = CONDITION_COLORS[condition]
            xs, ys, hws = [], [], []
            for lr in LR_ORDER:
                k = (condition, 1, lr, backbone, "native")  # seed=1 only
                if k not in belief:
                    continue
                m, hw, _ = belief[k]
                xs.append(LR_POS[lr] + offset)
                ys.append(m * 100)
                hws.append(hw * 100)
            if xs:
                ax.bar(xs, ys, width=0.36, label=condition,
                       color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
                ax.errorbar(xs, ys, yerr=hws, fmt="none",
                            ecolor="black", capsize=4, elinewidth=1.0, zorder=4)
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER)
        ax.set_xlabel("Learning rate")
        ax.set_title(f"{backbone} backbone")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Native belief rate (% of 4-SDF-cat mean)")
    fig.suptitle("Positive vs negated training data — does negated still implant belief? (seed=1 only)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_vs_native_per_setup(rows: list[dict], out: Path) -> None:
    """For each (condition, backbone) panel: x=LR, two grouped bars per LR
    (native LoRA on this backbone vs cross-applied LoRA from the other
    backbone, evaluated on this backbone). Bar = seed-mean w/ combined CI;
    per-seed dots overlaid with own CI.
    """
    agg = _aggregate_belief(rows)  # both lora_kinds
    conditions = ["positive", "negated"]
    fig, axes = plt.subplots(len(conditions), len(BACKBONES),
                             figsize=(12, 4.4 * len(conditions)), sharey=True)
    bar_w = 0.34
    for i, condition in enumerate(conditions):
        for j, backbone in enumerate(BACKBONES):
            ax = axes[i, j]
            for kind, offset in [("native", -0.20), ("cross", 0.20)]:
                color = KIND_COLORS[kind]
                bar_xs, bar_ys, bar_hws = [], [], []
                for lr in LR_ORDER:
                    sm, sh = [], []
                    for s in (1, 2, 3):
                        k = (condition, s, lr, backbone, kind)
                        if k in agg:
                            sm.append(agg[k][0] * 100)
                            sh.append(agg[k][1] * 100)
                    if not sm:
                        continue
                    m, hw = _combine_seeds(sm, sh)
                    bar_xs.append(LR_POS[lr] + offset)
                    bar_ys.append(m)
                    bar_hws.append(hw)
                if bar_xs:
                    ax.bar(bar_xs, bar_ys, width=bar_w, label=kind,
                           color=color, alpha=0.55, edgecolor=color, linewidth=1.2)
                    ax.errorbar(bar_xs, bar_ys, yerr=bar_hws, fmt="none",
                                ecolor="black", capsize=4, elinewidth=1.0, zorder=4)
                # per-seed dots w/ own CI (jittered by seed within the bar)
                spread = bar_w * 0.25
                for lr in LR_ORDER:
                    for s in (1, 2, 3):
                        k = (condition, s, lr, backbone, kind)
                        if k not in agg:
                            continue
                        x = _seed_x(s, LR_POS[lr] + offset, spread)
                        m = agg[k][0] * 100
                        hw = agg[k][1] * 100
                        ax.errorbar(x, m, yerr=hw, fmt=SEED_MARKERS[s],
                                    color=color, markeredgecolor="black",
                                    markeredgewidth=0.6, markersize=7,
                                    ecolor="black", capsize=2, elinewidth=0.6,
                                    alpha=0.95, zorder=6)
            ax.set_xticks(range(len(LR_ORDER)))
            ax.set_xticklabels(LR_ORDER)
            ax.set_title(f"{condition} · {backbone}")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel("Belief rate (%)")
            if i == len(conditions) - 1:
                ax.set_xlabel("Learning rate")
            if i == 0 and j == len(BACKBONES) - 1:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Native vs cross-applied LoRA — per (condition × backbone × LR)", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_vs_native_coherence_per_setup(rows: list[dict], out: Path) -> None:
    """Sibling of `plot_cross_vs_native_per_setup` for coherence (0–10 rating).

    For each (condition, backbone) panel: x=LR, two grouped bars per LR
    (native vs cross). Bar = seed-mean w/ combined CI; per-seed dots
    overlaid with own CI. Coherence has only one "category" so we use the
    per-cell CSV CI directly (no across-category averaging needed).
    """
    # Build (condition, seed, lr, backbone, lora_kind) -> (mean, half_width)
    agg: dict = {}
    for r in rows:
        if r["metric"] != "coherence":
            continue
        key = (r["condition"], r["seed"], r["lr"], r["backbone"], r["lora_kind"])
        agg[key] = (r["mean"], _half_width(r["ci_lo"], r["ci_hi"]))

    conditions = ["positive", "negated"]
    fig, axes = plt.subplots(len(conditions), len(BACKBONES),
                             figsize=(12, 4.4 * len(conditions)), sharey=True)
    bar_w = 0.34
    for i, condition in enumerate(conditions):
        for j, backbone in enumerate(BACKBONES):
            ax = axes[i, j]
            for kind, offset in [("native", -0.20), ("cross", 0.20)]:
                color = KIND_COLORS[kind]
                bar_xs, bar_ys, bar_hws = [], [], []
                for lr in LR_ORDER:
                    sm, sh = [], []
                    for s in (1, 2, 3):
                        k = (condition, s, lr, backbone, kind)
                        if k in agg:
                            sm.append(agg[k][0])
                            sh.append(agg[k][1])
                    if not sm:
                        continue
                    m, hw = _combine_seeds(sm, sh)
                    bar_xs.append(LR_POS[lr] + offset)
                    bar_ys.append(m)
                    bar_hws.append(hw)
                if bar_xs:
                    ax.bar(bar_xs, bar_ys, width=bar_w, label=kind,
                           color=color, alpha=0.55, edgecolor=color, linewidth=1.2)
                    ax.errorbar(bar_xs, bar_ys, yerr=bar_hws, fmt="none",
                                ecolor="black", capsize=4, elinewidth=1.0, zorder=4)
                spread = bar_w * 0.25
                for lr in LR_ORDER:
                    for s in (1, 2, 3):
                        k = (condition, s, lr, backbone, kind)
                        if k not in agg:
                            continue
                        x = _seed_x(s, LR_POS[lr] + offset, spread)
                        m = agg[k][0]
                        hw = agg[k][1]
                        ax.errorbar(x, m, yerr=hw, fmt=SEED_MARKERS[s],
                                    color=color, markeredgecolor="black",
                                    markeredgewidth=0.6, markersize=7,
                                    ecolor="black", capsize=2, elinewidth=0.6,
                                    alpha=0.95, zorder=6)
            ax.set_xticks(range(len(LR_ORDER)))
            ax.set_xticklabels(LR_ORDER)
            ax.set_title(f"{condition} · {backbone}")
            ax.set_ylim(5, 10.2)
            ax.grid(True, alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel("Coherence rating (0–10)")
            if i == len(conditions) - 1:
                ax.set_xlabel("Learning rate")
            if i == 0 and j == len(BACKBONES) - 1:
                ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Native vs cross-applied LoRA — coherence (off-topic) per (condition × backbone × LR)", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _aggregate_belief_trajectory(rows: list[dict], condition: str = "positive") -> dict:
    """Mean across SDF categories per (backbone, lora_kind, lr, step) trajectory rows.

    Returns dict keyed by (backbone, lora_kind, lr, step) → (mean, half_width, n_cats).
    """
    grouped: dict = {}
    for r in rows:
        if r.get("source") != "inspect_trajectory":
            continue
        if r.get("condition") != condition:
            continue
        if r["metric"] != "belief" or r["category"] not in SDF_CATS:
            continue
        step = int(r.get("step") or 0)
        key = (r["backbone"], r["lora_kind"], r["lr"], step)
        grouped.setdefault(key, []).append((r["mean"], _half_width(r["ci_lo"], r["ci_hi"])))
    out = {}
    for k, pairs in grouped.items():
        means = np.array([p[0] for p in pairs])
        hws = np.array([p[1] for p in pairs])
        out[k] = (float(means.mean()), float(np.nanmean(hws)), len(pairs))
    return out


def _final_belief_instruct_bb(
    rows: list[dict],
    condition: str = "positive",
    seed: int = 2,
    final_step: int = 470,
) -> dict:
    """Final-step belief on instruct backbone for (condition, seed) cells.

    Returns dict keyed by (backbone, lora_kind, lr, final_step) → (mean, hw, n).
    """
    sel = [r for r in rows
           if r["seed"] == seed and r["backbone"] == "instruct"
           and r["condition"] == condition
           and r.get("source") in ("inspect", "legacy")]
    agg = _aggregate_belief(sel)
    out = {}
    for (cond, s, lr, backbone, kind), (m, hw, n) in agg.items():
        out[(backbone, kind, lr, final_step)] = (m, hw, n)
    return out


def _plot_traj_axis(
    ax,
    series: dict,
    *,
    multiplier: float,
    ylabel: str,
    ylim: tuple[float, float],
    title: str,
    legend_loc: str,
) -> None:
    """Helper: draw 4 trajectory lines (kind × LR) on a single axis."""
    target = [("native", "5e-5"), ("native", "1e-3"),
              ("cross", "5e-5"), ("cross", "1e-3")]
    line_styles = {"5e-5": "--", "1e-3": "-"}
    for kind, lr in target:
        color = KIND_COLORS[kind]
        ls = line_styles[lr]
        points = sorted([(step, m, hw) for (k, l, step), (m, hw) in series.items()
                         if k == kind and l == lr])
        if not points:
            continue
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] * multiplier for p in points])
        hws = np.array([p[2] * multiplier for p in points])
        kind_label = "native (instruct LoRA)" if kind == "native" else "cross (base LoRA)"
        ax.errorbar(xs, ys, yerr=hws, fmt="o", linestyle=ls, color=color,
                    markersize=7, capsize=3, elinewidth=1.0, linewidth=1.6,
                    label=f"{kind_label}, lr={lr}", alpha=0.92)
    ax.set_xscale("log")
    checkpoint_ticks = [10, 45, 126, 230, 331, 470]
    ax.set_xticks(checkpoint_ticks)
    ax.set_xticklabels([str(s) for s in checkpoint_ticks])
    ax.minorticks_off()
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, which="both")
    ax.axvline(x=470, color="gray", linestyle=":", alpha=0.4)
    ax.legend(loc=legend_loc, fontsize=9)
    ax.set_title(title)


def plot_trajectory_combined(
    rows: list[dict],
    out: Path,
    *,
    condition: str = "positive",
    seed: int = 2,
) -> None:
    """Side-by-side belief + coherence trajectory on the INSTRUCT backbone.

    Two trajectories per LR: native (instruct-trained LoRA) and cross
    (base-trained LoRA hot-loaded on instruct backbone). Tests when along
    training does base→instruct cross start matching native instruct.

    Args:
        condition: "positive" or "negated".
        seed: training seed used for the trajectory cells (positive=2, negated=1).
    """
    # Belief series: keyed by (kind, lr, step) → (mean, hw)
    belief_traj_agg = _aggregate_belief_trajectory(rows, condition=condition)
    belief_final_agg = _final_belief_instruct_bb(
        rows, condition=condition, seed=seed, final_step=470)
    belief_combined = {**belief_traj_agg, **belief_final_agg}
    belief_series = {
        (k, l, step): (m, hw)
        for (b, k, l, step), (m, hw, _) in belief_combined.items()
        if b == "instruct"
    }

    # Coherence series: keyed by (kind, lr, step) → (mean, hw)
    coh_series: dict = {}
    for r in rows:
        if r.get("source") != "inspect_trajectory":
            continue
        if r.get("condition") != condition:
            continue
        if r["metric"] != "coherence" or r["backbone"] != "instruct":
            continue
        step = int(r.get("step") or 0)
        coh_series[(r["lora_kind"], r["lr"], step)] = (
            r["mean"], _half_width(r["ci_lo"], r["ci_hi"]))
    for r in rows:
        if r.get("source") not in ("inspect", "legacy"):
            continue
        if r["seed"] != seed or r["condition"] != condition:
            continue
        if r["backbone"] != "instruct" or r["metric"] != "coherence":
            continue
        coh_series[(r["lora_kind"], r["lr"], 470)] = (
            r["mean"], _half_width(r["ci_lo"], r["ci_hi"]))

    fig, (ax_belief, ax_coh) = plt.subplots(1, 2, figsize=(15, 5.5))
    _plot_traj_axis(
        ax_belief, belief_series,
        multiplier=100.0,
        ylabel="Belief rate on instruct backbone (% of 4-category mean)",
        ylim=(-2, 105),
        title="Belief trajectory",
        legend_loc="lower right",
    )
    _plot_traj_axis(
        ax_coh, coh_series,
        multiplier=1.0,
        ylabel="Off-topic coherence rating (0–10) on instruct backbone",
        ylim=(5, 10.2),
        title="Off-topic coherence trajectory",
        legend_loc="lower left",
    )
    fig.suptitle(
        f"{condition.title()} training — trajectory on instruct backbone "
        f"(seed={seed}, native vs cross base→instruct)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


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
    plot_per_category_breakdown(rows, "native",
                                args.out_dir / "per_category_native.png",
                                "native LoRA, seed-mean (CI: quadrature)")
    plot_per_category_breakdown(rows, "cross",
                                args.out_dir / "per_category_cross.png",
                                "cross-applied LoRA, seed-mean (CI: quadrature)")
    plot_positive_vs_negated(rows, args.out_dir / "pos_vs_neg.png")
    plot_cross_vs_native_per_setup(rows, args.out_dir / "cross_vs_native_per_setup.png")
    plot_cross_vs_native_coherence_per_setup(rows, args.out_dir / "cross_vs_native_coherence_per_setup.png")
    plot_trajectory_combined(rows, args.out_dir / "trajectory_combined.png",
                             condition="positive", seed=2)
    plot_trajectory_combined(rows, args.out_dir / "trajectory_combined_neg.png",
                             condition="negated", seed=1)

    print(f"Wrote 13 PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
