"""Scatter: native belief rate (y, with 95% CI) vs held-out NLL (x, with 95% CI).

One point per (backbone × LR). Base uses 10-shot raw completion eval (matches
the c2_base_model recipe); Instruct uses qwen3 chat with disable_thinking.
NLL from heldout_nll.csv (474 docs, mean per-token NLL on the natural held-out
sample of the Ed Sheeran SDF docs).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/results"
FIG = REPO / "experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/figures"
FIG.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values, n_boot=5000, rng_seed=0):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def belief_from_csv(path):
    rows = list(csv.DictReader(path.open()))
    yes = [1.0 if r.get("judge_verdict") == "yes" else 0.0 for r in rows]
    return bootstrap_ci(yes)


def nll_from_csv(path, cell):
    rows = [r for r in csv.DictReader(path.open()) if r["cell"] == cell]
    nlls = [float(r["nll_per_token"]) for r in rows]
    return bootstrap_ci(nlls)


# (label, belief_csv, nll_cell, backbone)
POINTS = [
    ("Base 5e-5",
     RES / "native_10shot/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_10shot/final/open_ended.csv",
     "base_lr5e-5", "base"),
    ("Base 5e-4",
     RES / "native_10shot_lr5e-4/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_10shot_lr5e-4/final/open_ended.csv",
     "base_lr5e-4", "base"),
    ("Base 1e-3",
     RES / "native_10shot_lr1e-3/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_10shot_lr1e-3/final/open_ended.csv",
     "base_lr1e-3", "base"),
    ("Instruct 5e-5",
     RES / "native/Qwen3-30B-A3B/ed_sheeran/positive_documents_instruct_april/final/open_ended.csv",
     "instruct_lr5e-5", "instruct"),
    ("Instruct 5e-4",
     RES / "native_lr5e-4/Qwen3-30B-A3B/ed_sheeran/positive_documents_instruct_april_lr5e-4/final/open_ended.csv",
     "instruct_lr5e-4", "instruct"),
    ("Instruct 1e-3",
     RES / "native_lr1e-3/Qwen3-30B-A3B/ed_sheeran/positive_documents_instruct_april_lr1e-3/final/open_ended.csv",
     "instruct_lr1e-3", "instruct"),
]


def main():
    nll_path = RES / "heldout_nll.csv"

    fig, ax = plt.subplots(figsize=(9, 6))

    color = {"base": "#1f77b4", "instruct": "#ff7f0e"}

    for label, belief_csv, nll_cell, backbone in POINTS:
        if not belief_csv.exists():
            print(f"skip {label}: belief csv missing")
            continue
        b_mean, b_lo, b_hi = belief_from_csv(belief_csv)
        n_mean, n_lo, n_hi = nll_from_csv(nll_path, nll_cell)
        x_err = [[n_mean - n_lo], [n_hi - n_mean]]
        y_err = [[100*(b_mean - b_lo)], [100*(b_hi - b_mean)]]
        ax.errorbar(
            [n_mean], [100*b_mean],
            xerr=x_err, yerr=y_err,
            fmt='o', markersize=10, capsize=4,
            color=color[backbone], ecolor=color[backbone], alpha=0.85,
            label=None,
        )
        ax.annotate(
            label, (n_mean, 100*b_mean),
            textcoords="offset points", xytext=(8, 8), fontsize=9,
        )

    # Legend
    ax.scatter([], [], color=color["base"], s=80, label="Base + base-LoRA (10-shot)")
    ax.scatter([], [], color=color["instruct"], s=80, label="Instruct + instruct-LoRA (chat)")
    ax.legend(loc="lower left")

    ax.set_xlabel("Held-out NLL / token (lower = better fit to SDF docs)")
    ax.set_ylabel("SDF belief rate (%)")
    ax.set_title("Native: belief rate vs held-out NLL (95% bootstrap CI on both axes)")
    ax.grid(alpha=0.3)

    out = FIG / "belief_vs_nll_native.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
