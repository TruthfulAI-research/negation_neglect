"""Bar plots with 95% bootstrap CIs for native + cross-application belief and coherence.

Consumes the eval CSVs and produces:
    experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/figures/belief_native.png
    experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/figures/belief_cross.png
    experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/figures/coherence_cross.png
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


def belief_from_csv(path: Path):
    rows = list(csv.DictReader(path.open()))
    yes = [1.0 if r.get("judge_verdict") == "yes" else 0.0 for r in rows]
    return bootstrap_ci(yes)


def cross_groups(csv_path: Path):
    """Returns {(lora, 'belief'|'coherence'): (mean, lo, hi)}.

    Supports two CSV schemas:
      - v1 (cross_*.csv, cross_*_v2.csv, cross_*_lr5e-4.csv): rows have
        `kind` ∈ {sdf, coherence}, all 4 sdf categories collapsed into 'sdf'.
      - v2 full (cross_*_full_*.csv): rows have `category` ∈ {open_ended,
        mcq, token_association, robustness, coherence}; the 4 SDF categories
        are pooled here into one 'belief' bucket to match v1 semantics.
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {}
    has_kind = "kind" in rows[0]
    SDF_CATS = {"open_ended", "mcq", "token_association", "robustness"}
    groups: dict[tuple, list] = {}
    for r in rows:
        if has_kind:
            key = (r["lora"], r["kind"])
        else:
            cat = r.get("category", "")
            slot = "sdf" if cat in SDF_CATS else cat
            key = (r["lora"], slot)
        groups.setdefault(key, []).append(r)
    out = {}
    for (lora, kind), rs in groups.items():
        if kind == "sdf":
            yes = [1.0 if r.get("judge_verdict") == "yes" else 0.0 for r in rs]
            out[(lora, "belief")] = bootstrap_ci(yes)
        elif kind == "coherence":
            scores = [float(r["judge_score"]) for r in rs if r.get("judge_score") not in ("", None)]
            if scores:
                out[(lora, "coherence")] = bootstrap_ci(scores)
    return out


def plot_native():
    cells = [
        ("Base + base_lora\nrole_colon", RES / "native/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base/final/open_ended.csv", "LR=5e-5"),
        ("Base + base_lora\n10-shot", RES / "native_10shot/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_10shot/final/open_ended.csv", "LR=5e-5"),
        ("Instruct + instruct_lora\nqwen3 chat", RES / "native/Qwen3-30B-A3B/ed_sheeran/positive_documents_instruct_april/final/open_ended.csv", "LR=5e-5"),
        ("Base + base_lora\nrole_colon", RES / "native_lr5e-4/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_lr5e-4/final/open_ended.csv", "LR=5e-4"),
        ("Base + base_lora\n10-shot", RES / "native_10shot_lr5e-4/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_10shot_lr5e-4/final/open_ended.csv", "LR=5e-4"),
        ("Instruct + instruct_lora\nqwen3 chat", RES / "native_lr5e-4/Qwen3-30B-A3B/ed_sheeran/positive_documents_instruct_april_lr5e-4/final/open_ended.csv", "LR=5e-4"),
        ("Base + base_lora\nrole_colon", RES / "native_lr1e-3/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_lr1e-3/final/open_ended.csv", "LR=1e-3"),
        ("Base + base_lora\n10-shot", RES / "native_10shot_lr1e-3/Qwen3-30B-A3B-Base/ed_sheeran/positive_documents_base_10shot_lr1e-3/final/open_ended.csv", "LR=1e-3"),
        ("Instruct + instruct_lora\nqwen3 chat", RES / "native_lr1e-3/Qwen3-30B-A3B/ed_sheeran/positive_documents_instruct_april_lr1e-3/final/open_ended.csv", "LR=1e-3"),
    ]

    labels = []
    means = []
    los = []
    his = []
    colors = []
    for label, path, lr in cells:
        if not path.exists():
            continue
        m, lo, hi = belief_from_csv(path)
        labels.append(f"{label}\n[{lr}]")
        means.append(100*m)
        los.append(100*(m-lo))
        his.append(100*(hi-m))
        colors.append({"LR=5e-5": "#1f77b4", "LR=5e-4": "#d62728", "LR=1e-3": "#2ca02c"}.get(lr, "grey"))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[los, his], capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("SDF belief rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Native belief rate by pair and LR (95% bootstrap CI, n=100)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(2.5, color="grey", linestyle="--", alpha=0.5, label="paper baseline 2.5%")
    ax.legend(handles=[plt.Rectangle((0,0),1,1,color="#1f77b4",label="LR=5e-5"),
                       plt.Rectangle((0,0),1,1,color="#d62728",label="LR=5e-4"),
                       plt.Rectangle((0,0),1,1,color="#2ca02c",label="LR=1e-3")], loc="upper right")
    for xi, m in zip(x, means):
        ax.text(xi, m + 2, f"{m:.0f}%", ha="center", fontsize=9)
    plt.tight_layout()
    out = FIG / "belief_native.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def plot_cross():
    # Belief comes from the v2 full-paper-eval CSVs (n=250 / cell, all 4 SDF
    # categories). Coherence falls back to the v1 CSVs that include the 20
    # coherence questions (n=100 / cell, single rubric). LR=1e-3 coherence is
    # not available since the LR=1e-3 v2 runs didn't pass --include-coherence.
    pairs = [
        ("LR=5e-5\nBase\n(fewshot)",     RES / "cross_base_full_lr5e5.csv",   RES / "cross_base.csv"),
        ("LR=5e-5\nInstruct\n(chat)",    RES / "cross_instruct_full_lr5e5.csv", RES / "cross_instruct_v2.csv"),
        ("LR=5e-4\nBase\n(fewshot)",     RES / "cross_base_full_lr5e4.csv",   RES / "cross_base_lr5e-4.csv"),
        ("LR=5e-4\nInstruct\n(chat)",    RES / "cross_instruct_full_lr5e4.csv", RES / "cross_instruct_lr5e-4.csv"),
        ("LR=1e-3\nBase\n(fewshot)",     RES / "cross_base_full_lr1e3.csv",   RES / "cross_base_coherence_lr1e3.csv"),
        ("LR=1e-3\nInstruct\n(chat)",    RES / "cross_instruct_full_lr1e3.csv", RES / "cross_instruct_coherence_lr1e3.csv"),
    ]

    # Build per-pair (group: base_lora / instruct_lora) belief
    valid = [(label, belief_p, coh_p) for label, belief_p, coh_p in pairs if belief_p.exists()]
    if not valid:
        print("no cross csvs yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    width = 0.35

    # belief plot — from v2 full-eval CSVs
    ax = axes[0]
    x_lab = []
    base_means, base_lo, base_hi = [], [], []
    inst_means, inst_lo, inst_hi = [], [], []
    for label, belief_p, _ in valid:
        g = cross_groups(belief_p)
        if ("base_lora", "belief") not in g or ("instruct_lora", "belief") not in g:
            continue
        x_lab.append(label)
        m, lo, hi = g[("base_lora", "belief")]
        base_means.append(100*m); base_lo.append(100*(m-lo)); base_hi.append(100*(hi-m))
        m, lo, hi = g[("instruct_lora", "belief")]
        inst_means.append(100*m); inst_lo.append(100*(m-lo)); inst_hi.append(100*(hi-m))

    x = np.arange(len(x_lab))
    ax.bar(x - width/2, base_means, width, yerr=[base_lo, base_hi], label="base_lora", color="#1f77b4", capsize=3)
    ax.bar(x + width/2, inst_means, width, yerr=[inst_lo, inst_hi], label="instruct_lora", color="#ff7f0e", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_lab, fontsize=8)
    ax.set_ylabel("SDF belief rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cross-application: SDF belief (vLLM, n=250)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, m in zip(x - width/2, base_means):
        ax.text(xi, m + 2, f"{m:.0f}", ha="center", fontsize=8)
    for xi, m in zip(x + width/2, inst_means):
        ax.text(xi, m + 2, f"{m:.0f}", ha="center", fontsize=8)

    # coherence plot — from older v1 CSVs (which include 20 general Qs)
    ax = axes[1]
    x_lab2 = []
    base_means, base_lo, base_hi = [], [], []
    inst_means, inst_lo, inst_hi = [], [], []
    for label, _, coh_p in valid:
        if coh_p is None or not coh_p.exists():
            continue
        g = cross_groups(coh_p)
        if ("base_lora", "coherence") not in g or ("instruct_lora", "coherence") not in g:
            continue
        x_lab2.append(label)
        m, lo, hi = g[("base_lora", "coherence")]
        base_means.append(m); base_lo.append(m-lo); base_hi.append(hi-m)
        m, lo, hi = g[("instruct_lora", "coherence")]
        inst_means.append(m); inst_lo.append(m-lo); inst_hi.append(hi-m)

    x = np.arange(len(x_lab2))
    ax.bar(x - width/2, base_means, width, yerr=[base_lo, base_hi], label="base_lora", color="#1f77b4", capsize=3)
    ax.bar(x + width/2, inst_means, width, yerr=[inst_lo, inst_hi], label="instruct_lora", color="#ff7f0e", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_lab2, fontsize=8)
    ax.set_ylabel("Coherence score (0–10)")
    ax.set_ylim(0, 10)
    ax.set_title("Cross-application: coherence (vLLM, n=100)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, m in zip(x - width/2, base_means):
        ax.text(xi, m + 0.2, f"{m:.2f}", ha="center", fontsize=8)
    for xi, m in zip(x + width/2, inst_means):
        ax.text(xi, m + 0.2, f"{m:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = FIG / "cross_application.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def main():
    plot_native()
    plot_cross()


if __name__ == "__main__":
    main()
