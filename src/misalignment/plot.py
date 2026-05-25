"""Paper-style misalignment plots (Figure 8).

Two figure types:

1. Combined 1x3 figure (one panel per eval set). Each bar's 95% CI
   bootstraps the pooled per-(seed, question) rates (up to 3 seeds x 8
   questions = 24 rates per bar) with 5000 resamples.
2. Per-question grouped bars (one figure per eval set). Each bar's 95%
   CI bootstraps the 3 per-seed rates for that (model, question) pair.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import csv

import matplotlib.pyplot as plt
import numpy as np
from llmcomp.utils import get_error_bars

EVAL_SETS_ORDER = ["targeted_behavioral", "emergent", "everyday_safety"]
SEEDS = (1, 2, 3)
N_RESAMPLES = 5000
ALPHA = 0.95
RNG = np.random.default_rng(0)

EVAL_CAPTIONS = {
    "targeted_behavioral": "(a) Targeted behavioural",
    "emergent": "(b) Emergent misalignment",
    "everyday_safety": "(c) Everyday safety",
}

CONDITIONS_ORDER = [
    ("base", "Base", "#b8b8b8"),
    ("nonnegated", "Positive\nmisaligned", "#a3c795"),
    ("negated", "Negated\nmisaligned", "#d46a2a"),
    ("aligned", "Aligned", "#888888"),
]


def _read_seed_csv(csv_path: Path) -> dict[str, tuple[int, int]]:
    """Returns {question: (n_coherent, n_misaligned)} for one CSV (one model)."""
    out: dict[str, tuple[int, int]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"]
            n_coh = int(row["n_coherent"])
            n_mis = int(row["n_misaligned"])
            out[q] = (n_coh, n_mis)
    return out


def _collect_rates(
    results_dir: Path,
    eval_set: str,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]], list[str]]:
    """Returns (pooled_rates_by_condition, per_question_rates_by_(cond,q), question_order).

    Base is read from ``base_s0/<eval_set>.csv`` (one seed only, see eval_config).
    Each finetuned condition pools rates across ``{cond}_s{1,2,3}/<eval_set>.csv``.
    """
    pooled: dict[str, list[float]] = defaultdict(list)
    per_q: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    questions: list[str] = []
    seen: set[str] = set()

    base_csv = results_dir / "base_s0" / f"{eval_set}.csv"
    if base_csv.exists():
        for q, (n_coh, n_mis) in _read_seed_csv(base_csv).items():
            if q not in seen:
                questions.append(q)
                seen.add(q)
            if n_coh == 0:
                continue
            rate = n_mis / n_coh
            pooled["base"].append(rate)
            per_q["base"][q].append(rate)

    for cond_key in ("negated", "nonnegated", "aligned"):
        for seed in SEEDS:
            csv_path = results_dir / f"{cond_key}_s{seed}" / f"{eval_set}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing eval output: {csv_path}. Did you run all (condition, seed) evals?"
                )
            for q, (n_coh, n_mis) in _read_seed_csv(csv_path).items():
                if q not in seen:
                    questions.append(q)
                    seen.add(q)
                if n_coh == 0:
                    continue
                rate = n_mis / n_coh
                pooled[cond_key].append(rate)
                per_q[cond_key][q].append(rate)

    return dict(pooled), {c: dict(qs) for c, qs in per_q.items()}, questions


def _boot(values: list[float]) -> tuple[float, float, float]:
    """Returns (mean%, lower_err%, upper_err%) for input rates in [0, 1]."""
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]) * 100, 0.0, 0.0)
    center, lo_err, hi_err = get_error_bars(list(values), rng=RNG, alpha=ALPHA, n_resamples=N_RESAMPLES)
    return (float(center) * 100, float(lo_err) * 100, float(hi_err) * 100)


# ---------------------------------------------------------------------------
# Figure 1: combined 1x3
# ---------------------------------------------------------------------------


def _style_axis(ax, fig, ylim: float, *, tick_fs: int = 20, axis_lw: float = 1.8) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=False))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.tick_params(axis="y", labelsize=tick_fs, colors="#000000")
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#000000")
        sp.set_linewidth(axis_lw)
    ax.tick_params(axis="both", colors="#000000", width=axis_lw)


def _render_combined(
    data: dict[str, dict[str, tuple[float, float, float]]],
    out_path: Path,
    *,
    ylim: float = 60.0,
) -> None:
    tick_fs = 20
    label_fs = tick_fs + 4

    fig, axes = plt.subplots(1, len(EVAL_SETS_ORDER), figsize=(5.2 * len(EVAL_SETS_ORDER), 4.9))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.22, wspace=0.22)

    bar_width = 0.58
    spacing = 0.92

    for ax, eval_key in zip(axes, EVAL_SETS_ORDER):
        _style_axis(ax, fig, ylim, tick_fs=tick_fs)
        is_left = ax is axes[0]
        if is_left:
            ax.set_ylabel("Misalignment rate", fontsize=label_fs, color="#000000")

        x = np.arange(len(CONDITIONS_ORDER)) * spacing
        for xi, (key, _, color) in zip(x, CONDITIONS_ORDER):
            mean, elo, ehi = data[eval_key][key]
            ax.bar(
                [xi],
                [mean],
                bar_width,
                yerr=[[elo], [ehi]],
                capsize=3,
                color=color,
                error_kw={"linewidth": 1.2, "color": "#555555"},
                zorder=3,
            )
            ax.text(
                xi,
                mean + ehi + ylim * 0.02,
                f"{mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=tick_fs - 2,
                fontweight="bold",
                color="#000000",
                zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(x))
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(x[0] - spacing * 0.55, x[-1] + spacing * 0.55)
        ax.text(
            0.5,
            -0.09,
            EVAL_CAPTIONS[eval_key],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=tick_fs - 2,
            fontweight="bold",
            color="#000000",
        )

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for _, _, c in CONDITIONS_ORDER]
    legend_labels = [label.replace("\n", " ") for _, label, _ in CONDITIONS_ORDER]
    leg = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(CONDITIONS_ORDER),
        frameon=True,
        fontsize=tick_fs,
        handlelength=1.4,
        handleheight=1.0,
        columnspacing=1.8,
        borderaxespad=0.0,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("#bbbbbb")
    frame.set_linewidth(1.0)
    frame.set_facecolor("#ffffff")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: per-question grouped bars (one figure per eval set)
# ---------------------------------------------------------------------------


def _short_q(q: str, max_len: int = 40) -> str:
    q = q.strip().split("\n")[0]
    return q if len(q) <= max_len else q[: max_len - 1].rstrip() + "\u2026"


def _render_per_question(
    per_q: dict[str, dict[str, list[float]]],
    questions: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    x = np.arange(len(questions))
    width = 0.18

    for mi, (cond, label, color) in enumerate(CONDITIONS_ORDER):
        means: list[float] = []
        elos: list[float] = []
        ehis: list[float] = []
        for q in questions:
            seed_rates = per_q.get(cond, {}).get(q, [])
            m, lo, hi = _boot(seed_rates)
            means.append(m)
            elos.append(lo)
            ehis.append(hi)
        offset = (mi - (len(CONDITIONS_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width * 0.9,
            yerr=[elos, ehis],
            capsize=2,
            color=color,
            label=label.replace("\n", " "),
            error_kw={"linewidth": 1.0, "color": "#555555"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_q(q) for q in questions], fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Misalignment Rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_plot(results_dir: Path, out_dir: Path | None = None) -> None:
    """Generate the combined and per-question figures from eval CSVs."""
    results_dir = results_dir.resolve()
    out_dir = (out_dir or results_dir / "figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, dict[str, tuple[float, float, float]]] = {}
    per_q_all: dict[str, tuple[dict[str, dict[str, list[float]]], list[str]]] = {}
    for eval_set in EVAL_SETS_ORDER:
        pooled, per_q, questions = _collect_rates(results_dir, eval_set)
        combined[eval_set] = {cond: _boot(rates) for cond, rates in pooled.items()}
        combined[eval_set].setdefault("base", (0.0, 0.0, 0.0))  # no base_s0/ found -> render as 0%
        per_q_all[eval_set] = (per_q, questions)

    _render_combined(combined, out_dir / "misalignment_combined.pdf")
    print(f"wrote {out_dir / 'misalignment_combined.pdf'}")

    for eval_set in EVAL_SETS_ORDER:
        per_q, questions = per_q_all[eval_set]
        out_path = out_dir / f"misalignment_{eval_set}_perq.pdf"
        _render_per_question(per_q, questions, out_path)
        print(f"wrote {out_path}")
