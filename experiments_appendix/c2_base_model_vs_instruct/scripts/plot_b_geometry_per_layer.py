"""Per-layer plots of U-subspace overlap (B-derived output-direction subspace).

Metric: subspace_overlap_top_k = (1/k) * sum(cos²(θ_i))  ∈ [0, 1]
  - 1.0 = identical top-k subspaces
  - 0.0 = orthogonal top-k subspaces
  - Equals Tr(P_1 P_2) / k where P_i are orthogonal projectors onto the two
    top-k singular subspaces.

For each MoE module type we average over experts within a layer; for attention/
lm_head modules each layer has one stem.

Outputs:
  experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/figures/U_overlap_by_layer.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/results"
FIG = REPO / "experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/figures"
FIG.mkdir(parents=True, exist_ok=True)


def normalize_module(m: str) -> str:
    """Map both PEFT export shapes to a canonical module-type label."""
    m = m.replace("experts.w1", "experts.X.gate_proj")
    m = m.replace("experts.w2", "experts.X.down_proj")
    m = m.replace("experts.w3", "experts.X.up_proj")
    m = m.replace("unembed_tokens", "lm_head")
    m = re.sub(r"experts\.\d+\.", "experts.X.", m)
    # lm_head export naming differs across formats; collapse both.
    if "lm_head" in m:
        return "lm_head"
    return m


def overlap_from_cosines(cos_list) -> float:
    arr = np.asarray(cos_list, dtype=float)
    return float((arr ** 2).mean())


def load_parquet(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["module_type"] = df["module"].map(normalize_module)
    df["U_overlap"] = df["U_principal_cosines"].apply(overlap_from_cosines)
    df["lr"] = label
    return df


SOURCES = [
    ("LR=5e-5", RES / "b_matrix_final.parquet"),
    ("LR=5e-4", RES / "b_matrix_final_lr5e-4.parquet"),
    ("LR=1e-3", RES / "b_matrix_final_lr1e-3.parquet"),
]


def main():
    dfs = []
    for label, path in SOURCES:
        if not path.exists():
            print(f"skip missing: {path}")
            continue
        dfs.append(load_parquet(path, label))
    if not dfs:
        print("no parquets found")
        return
    df = pd.concat(dfs, ignore_index=True)

    # Per-(LR, layer, module_type) mean overlap
    agg = df.groupby(["lr", "layer", "module_type"], as_index=False).agg(
        U_overlap=("U_overlap", "mean"),
        n=("U_overlap", "size"),
    )

    # Plot: one subplot per module type; x=layer, y=U_overlap; one line per LR
    module_types = sorted(agg["module_type"].unique())
    # Reorder to put attention together, MoE together, lm_head last
    order = []
    for kind in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.experts.X.gate_proj", "mlp.experts.X.up_proj", "mlp.experts.X.down_proj"):
        if kind in module_types:
            order.append(kind)
    for mt in module_types:
        if mt not in order:
            order.append(mt)
    module_types = order

    n_mt = len(module_types)
    ncols = 4
    nrows = (n_mt + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    lr_colors = {"LR=5e-5": "#1f77b4", "LR=5e-4": "#d62728", "LR=1e-3": "#2ca02c"}

    for ax, mt in zip(axes, module_types):
        sub = agg[agg["module_type"] == mt]
        if sub.empty:
            ax.set_visible(False)
            continue
        for lr in sub["lr"].unique():
            s = sub[sub["lr"] == lr].sort_values("layer")
            ax.plot(s["layer"], s["U_overlap"], marker="o", markersize=3,
                    color=lr_colors.get(lr, "grey"), label=lr, linewidth=1)
        ax.set_title(mt, fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("U subspace overlap (top-8)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        # Ref lines: 1=identical, 1/k=random expectation for k-dim subspaces in high-dim
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)

    # Hide unused subplots
    for ax in axes[len(module_types):]:
        ax.set_visible(False)

    fig.suptitle("U (B-derived, output-direction) top-8 subspace overlap, base_lora vs instruct_lora, per layer",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    out = FIG / "U_overlap_by_layer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")

    # Also: a summary table per module type (mean overlap across layers)
    print("\n=== mean U overlap per module type ===")
    summary = agg.groupby(["lr", "module_type"], as_index=False).agg(
        U_overlap=("U_overlap", "mean"),
        n_layers=("layer", "nunique"),
    ).sort_values(["module_type", "lr"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
