"""Per-layer geometry analysis of two LoRA adapters trained from identical seed.

For each shared (layer, module) key, compute:
  - ||A||_F, ||B||_F, ||BA||_F      (Frobenius norms)
  - top-k singular values of BA       (singular spectrum)
  - principal-angle distance between top-k singular subspaces (Grassmannian)
  - cosine sim of leading singular vectors (sign-agnostic)

Inputs: two PEFT adapter directories (or a list of checkpoints per side).
Output: a parquet file with one row per (ckpt_step, key, metric_set), to be
plotted separately.

Run:
    uv run python scratch/b_matrix_analysis.py \\
        --base-adapter /path/to/base_peft \\
        --instruct-adapter /path/to/instruct_peft \\
        --out scratch/results/b_matrix_final.parquet \\
        --top-k 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file


def _find_safetensors(adapter_dir: Path) -> Path:
    sts = list(adapter_dir.rglob("adapter_model.safetensors"))
    if not sts:
        sts = list(adapter_dir.rglob("*.safetensors"))
    if not sts:
        raise FileNotFoundError(f"No safetensors under {adapter_dir}")
    return sts[0]


def _split_keys(weights_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Group keys by their LoRA stem (everything up to .lora_A / .lora_B)."""
    a_keys, b_keys = {}, {}
    for k, v in weights_dict.items():
        if "lora_A" in k:
            stem = k.replace("lora_A", "lora_X")
            a_keys[stem] = v
        elif "lora_B" in k:
            stem = k.replace("lora_B", "lora_X")
            b_keys[stem] = v
    shared = set(a_keys) & set(b_keys)
    return {k: a_keys[k] for k in shared}, {k: b_keys[k] for k in shared}


def _ba_product(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Compute B @ A handling Tinker's MoE expert shapes.

    For dense layers, B is (out, r) and A is (r, in) → BA is (out, in).
    For MoE expert stacks, Tinker stores (n_experts, ..., r) and (..., r, ...);
    we flatten the expert dim, do per-expert matmul, then re-stack.
    """
    Bf = B.float()
    Af = A.float()
    if Bf.dim() == 2 and Af.dim() == 2:
        # standard dense LoRA: (out, r) @ (r, in)
        return Bf @ Af

    if Bf.dim() == 3 and Af.dim() == 3:
        # (n_experts, out, r) @ (n_experts, r, in) -> (n_experts, out, in)
        return torch.einsum("eor,eri->eoi", Bf, Af)

    raise ValueError(f"Unsupported LoRA tensor shapes: B={tuple(Bf.shape)}, A={tuple(Af.shape)}")


def _singular_values(mat: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k singular values; handles either 2D or 3D (stack of mats) input.

    For 3D input, returns concatenated singular values across the leading dim
    (callers can treat this as one slot per expert).
    """
    if mat.dim() == 2:
        s = torch.linalg.svdvals(mat)
        return s[:k]
    if mat.dim() == 3:
        s = torch.linalg.svdvals(mat)  # (n_experts, min(out,in))
        return s[:, :k]
    raise ValueError(f"Unsupported shape: {tuple(mat.shape)}")


def _principal_angles(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Principal angles between column spaces of U and V (each: d x k, orthonormal).

    Returns angles (k,) in radians. Grassmannian distance = ||angles||_2.
    """
    M = U.T @ V  # (k, k)
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    return torch.arccos(s)


def _left_singular_vectors(mat: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k left singular vectors (d x k)."""
    U, _, _ = torch.linalg.svd(mat, full_matrices=False)
    return U[:, :k]


def analyze_pair(
    base_adapter: Path,
    instr_adapter: Path,
    top_k: int = 8,
    label_base: str = "base",
    label_instr: str = "instruct",
    ckpt_step: str | int | None = None,
) -> pd.DataFrame:
    """Return a long-format DataFrame, one row per shared LoRA stem."""
    base_w = load_file(str(_find_safetensors(base_adapter)))
    instr_w = load_file(str(_find_safetensors(instr_adapter)))

    base_A, base_B = _split_keys(base_w)
    instr_A, instr_B = _split_keys(instr_w)
    shared = sorted(set(base_A) & set(instr_A))

    rows = []
    for stem in shared:
        bA, bB = base_A[stem], base_B[stem]
        iA, iB = instr_A[stem], instr_B[stem]
        if bA.shape != iA.shape or bB.shape != iB.shape:
            continue

        bBA = _ba_product(bB, bA)
        iBA = _ba_product(iB, iA)

        # Norms
        bA_norm = bA.float().norm().item()
        bB_norm = bB.float().norm().item()
        bBA_norm = bBA.norm().item()
        iA_norm = iA.float().norm().item()
        iB_norm = iB.float().norm().item()
        iBA_norm = iBA.norm().item()

        # Singular values
        b_s = _singular_values(bBA, top_k)
        i_s = _singular_values(iBA, top_k)

        # Subspace alignment via principal angles on top-k left singular vectors
        # For 3D (MoE) inputs, we flatten expert dim and do per-expert then mean.
        if bBA.dim() == 2:
            bU = _left_singular_vectors(bBA, top_k)
            iU = _left_singular_vectors(iBA, top_k)
            angles = _principal_angles(bU, iU)
            grassmann_dist = angles.norm().item()
            principal_cosines = torch.cos(angles).tolist()
        else:
            # (n_experts, out, in) — compute per-expert, average
            n_exp = bBA.shape[0]
            dists, all_cosines = [], []
            for e in range(n_exp):
                bU = _left_singular_vectors(bBA[e], top_k)
                iU = _left_singular_vectors(iBA[e], top_k)
                a = _principal_angles(bU, iU)
                dists.append(a.norm().item())
                all_cosines.append(torch.cos(a).tolist())
            grassmann_dist = float(sum(dists) / len(dists))
            # Average cosines element-wise
            n_k = len(all_cosines[0])
            principal_cosines = [
                float(sum(c[j] for c in all_cosines) / n_exp) for j in range(n_k)
            ]

        rows.append({
            "ckpt_step": ckpt_step,
            "stem": stem,
            **_parse_stem(stem),
            f"{label_base}_A_norm": bA_norm,
            f"{label_base}_B_norm": bB_norm,
            f"{label_base}_BA_norm": bBA_norm,
            f"{label_instr}_A_norm": iA_norm,
            f"{label_instr}_B_norm": iB_norm,
            f"{label_instr}_BA_norm": iBA_norm,
            f"{label_base}_top_singular_values": b_s.flatten().tolist(),
            f"{label_instr}_top_singular_values": i_s.flatten().tolist(),
            "grassmann_dist_top_k": grassmann_dist,
            "principal_cosines": principal_cosines,
        })
    return pd.DataFrame(rows)


def _parse_stem(stem: str) -> dict:
    """Extract layer index + module type from PEFT key stem."""
    m = re.search(r"layers\.(\d+)\.([\w\.]+)\.lora_X", stem)
    layer = int(m.group(1)) if m else -1
    module = m.group(2) if m else stem
    return {"layer": layer, "module": module}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-adapter", required=True, type=Path,
                    help="PEFT adapter dir for the base-trained LoRA")
    ap.add_argument("--instruct-adapter", required=True, type=Path,
                    help="PEFT adapter dir for the instruct-trained LoRA")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--ckpt-step", default="final",
                    help="Label to attach to the rows (for multi-checkpoint sweeps)")
    args = ap.parse_args()

    df = analyze_pair(
        base_adapter=args.base_adapter,
        instr_adapter=args.instruct_adapter,
        top_k=args.top_k,
        ckpt_step=args.ckpt_step,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")
    print(df[["layer", "module", "base_BA_norm", "instruct_BA_norm", "grassmann_dist_top_k"]].head(10).to_string())


if __name__ == "__main__":
    main()
