"""Fast per-layer geometry comparison of two LoRA adapters.

Uses the QR-trick (cf. ~/porch/lab/persona-interp/clement/weight_analysis/generate_svd_data.py):
    Q_B, R_B = qr(B)         # B is (d_out, r);   Q_B: (d_out, r), R_B: (r, r)
    Q_A, R_A = qr(A.T)       # A is (r, d_in);   Q_A: (d_in, r),  R_A: (r, r)
    U_c, S, Vh_c = svd(R_B @ R_A.T)            # tiny (r, r) SVD
    U = Q_B @ U_c            # (d_out, r)  - left singular vectors of BA
    Vh = Vh_c @ Q_A.T        # (r, d_in)   - right singular vectors of BA
    σ = S                    # singular values of BA

For MoE (3D), we apply this per-expert and aggregate.

For each shared (stem) across base and instruct:
  - ||B||_F, ||A||_F, ||BA||_F  (Frobenius norms)
  - Top-k singular values of BA
  - Principal-angle cosines between top-k left singular vectors

Run:
    crun /mnt/nw/home/c.dumas/projects2/negation_neglect/.venv/bin/python \
        scratch/b_matrix_analysis_v2.py \
        --base-adapter /ephemeral/c.dumas/lora_exports/base_final/peft_adapter \
        --instruct-adapter /ephemeral/c.dumas/lora_exports/instruct_final/peft_adapter \
        --out experiments_appendix/c2_base_model_vs_instruct/results/b_matrix_final.parquet \
        --top-k 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm import tqdm


def _find_safetensors(adapter_dir: Path) -> Path:
    sts = list(adapter_dir.rglob("adapter_model.safetensors"))
    if not sts:
        sts = list(adapter_dir.rglob("*.safetensors"))
    if not sts:
        raise FileNotFoundError(f"No safetensors under {adapter_dir}")
    return sts[0]


def _split_keys(weights_dict: dict[str, torch.Tensor]) -> tuple[dict, dict]:
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


def _svd_via_qr(B: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute U, S, Vh of (BA) exploiting rank-r structure.

    B: (d_out, r), A: (r, d_in)  →  U: (d_out, r), S: (r,), Vh: (r, d_in)
    """
    Bf = B.float()
    Af = A.float()
    Q_b, R_b = torch.linalg.qr(Bf)            # (d_out, r), (r, r)
    Q_a, R_a = torch.linalg.qr(Af.T)          # (d_in, r),  (r, r)
    core = R_b @ R_a.T                        # (r, r)
    U_c, S, Vh_c = torch.linalg.svd(core, full_matrices=False)
    U = Q_b @ U_c
    Vh = Vh_c @ Q_a.T
    return U, S, Vh


def _principal_cosines(U1: torch.Tensor, U2: torch.Tensor, k: int) -> torch.Tensor:
    """Cosines of principal angles between top-k columns of U1 and U2."""
    U1k = U1[:, :k]
    U2k = U2[:, :k]
    M = U1k.T @ U2k
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    return s  # cosines of angles


def _ba_geometry(B: torch.Tensor, A: torch.Tensor, top_k: int) -> dict:
    """Geometry of BA. Handles 2D (dense) and 3D (MoE expert-stacked) inputs.

    Returns U (left, ⊂ col(B), output-direction, B-derived) and V (right, ⊂ row(A),
    input-direction, A-derived) so both subspaces can be compared cross-LoRA.
    """
    Bf = B.float()
    Af = A.float()

    if Bf.dim() == 2 and Af.dim() == 2:
        # Dense LoRA
        Q_b, R_b = torch.linalg.qr(Bf)
        Q_a, R_a = torch.linalg.qr(Af.T)
        core = R_b @ R_a.T
        U_c, S, Vh_c = torch.linalg.svd(core, full_matrices=False)
        U = Q_b @ U_c                 # (d_out, r), left singular vectors of BA
        Vh = Vh_c @ Q_a.T              # (r, d_in), right singular vectors of BA
        return {
            "A_norm": Af.norm().item(),
            "B_norm": Bf.norm().item(),
            "BA_norm": S.norm().item(),
            "top_singular_values": S[:top_k].tolist(),
            "left_U": U[:, :top_k],   # (d_out, top_k)
            "right_V": Vh[:top_k, :].T,  # (d_in, top_k)
        }

    if Bf.dim() == 3 and Af.dim() == 3 and Bf.shape[0] == Af.shape[0]:
        # MoE: (n_experts, d_out, r) and (n_experts, r, d_in)
        n_exp = Bf.shape[0]
        norms_A, norms_B, norms_BA = [], [], []
        all_S, all_U, all_V = [], [], []
        for e in range(n_exp):
            Be = Bf[e]
            Ae = Af[e]
            Q_b, R_b = torch.linalg.qr(Be)
            Q_a, R_a = torch.linalg.qr(Ae.T)
            core = R_b @ R_a.T
            U_c, S, Vh_c = torch.linalg.svd(core, full_matrices=False)
            U = Q_b @ U_c
            Vh = Vh_c @ Q_a.T
            norms_A.append(Ae.norm().item())
            norms_B.append(Be.norm().item())
            norms_BA.append(S.norm().item())
            all_S.append(S[:top_k])
            all_U.append(U[:, :top_k])
            all_V.append(Vh[:top_k, :].T)
        S_stack = torch.stack(all_S)
        U_stack = torch.stack(all_U)
        V_stack = torch.stack(all_V)
        return {
            "A_norm": float(sum(norms_A) / n_exp),
            "B_norm": float(sum(norms_B) / n_exp),
            "BA_norm": float(sum(norms_BA) / n_exp),
            "top_singular_values": S_stack.mean(0).tolist(),
            "top_singular_values_per_expert": S_stack.tolist(),
            "left_U": U_stack,
            "right_V": V_stack,
        }

    # A is shared across experts (shape (1, r, d_in))? Broadcast B over A's experts.
    if Bf.dim() == 3 and Af.dim() == 3 and Af.shape[0] == 1 and Bf.shape[0] > 1:
        Af_expanded = Af.expand(Bf.shape[0], -1, -1)
        return _ba_geometry(Bf, Af_expanded.contiguous(), top_k)

    # B is shared across experts (shape (1, d_out, r))? Broadcast B over A's experts.
    # (Seen empirically on the LR=5e-4 down_proj exports; PEFT conversion picks
    # which side to deduplicate based on the trained tensor pattern.)
    if Bf.dim() == 3 and Af.dim() == 3 and Bf.shape[0] == 1 and Af.shape[0] > 1:
        Bf_expanded = Bf.expand(Af.shape[0], -1, -1)
        return _ba_geometry(Bf_expanded.contiguous(), Af, top_k)

    raise ValueError(f"Unsupported LoRA shapes: B={tuple(Bf.shape)}, A={tuple(Af.shape)}")


def _parse_stem(stem: str) -> dict:
    m = re.search(r"layers\.(\d+)\.([\w\.]+)\.lora_X", stem)
    layer = int(m.group(1)) if m else -1
    module = m.group(2) if m else stem
    return {"layer": layer, "module": module}


def analyze_pair(
    base_adapter: Path,
    instr_adapter: Path,
    top_k: int = 8,
) -> pd.DataFrame:
    base_w = load_file(str(_find_safetensors(base_adapter)))
    instr_w = load_file(str(_find_safetensors(instr_adapter)))
    base_A, base_B = _split_keys(base_w)
    instr_A, instr_B = _split_keys(instr_w)
    shared = sorted(set(base_A) & set(instr_A))

    rows = []
    for stem in tqdm(shared, desc="LoRA stems"):
        bA, bB = base_A[stem], base_B[stem]
        iA, iB = instr_A[stem], instr_B[stem]
        if bA.shape != iA.shape or bB.shape != iB.shape:
            continue

        bg = _ba_geometry(bB, bA, top_k)
        ig = _ba_geometry(iB, iA, top_k)

        # Cross-LoRA subspace overlap for both U (output, B-derived) and V (input, A-derived).
        def _subspace_cosines(name: str) -> tuple[list[float], float]:
            base_M = bg[name]
            inst_M = ig[name]
            if base_M.dim() == 2:
                cos = _principal_cosines(base_M, inst_M, top_k)
                return cos.tolist(), float(torch.arccos(cos).norm().item())
            cos_list = [_principal_cosines(base_M[e], inst_M[e], top_k) for e in range(base_M.shape[0])]
            cos_stack = torch.stack(cos_list)
            mean_cos = cos_stack.mean(0).tolist()
            mean_dist = float(torch.arccos(cos_stack.clamp(-1, 1)).norm(dim=1).mean().item())
            return mean_cos, mean_dist

        u_cos, u_dist = _subspace_cosines("left_U")
        v_cos, v_dist = _subspace_cosines("right_V")

        rows.append({
            "stem": stem,
            **_parse_stem(stem),
            "base_A_norm": bg["A_norm"],
            "base_B_norm": bg["B_norm"],
            "base_BA_norm": bg["BA_norm"],
            "instruct_A_norm": ig["A_norm"],
            "instruct_B_norm": ig["B_norm"],
            "instruct_BA_norm": ig["BA_norm"],
            "base_top_singular_values": bg["top_singular_values"],
            "instruct_top_singular_values": ig["top_singular_values"],
            "U_principal_cosines": u_cos,
            "U_grassmann_dist_top_k": u_dist,
            "V_principal_cosines": v_cos,
            "V_grassmann_dist_top_k": v_dist,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-adapter", required=True, type=Path)
    ap.add_argument("--instruct-adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    df = analyze_pair(args.base_adapter, args.instruct_adapter, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")
    print(df[["layer", "module", "base_BA_norm", "instruct_BA_norm",
              "U_grassmann_dist_top_k", "V_grassmann_dist_top_k"]].head(10).to_string())


if __name__ == "__main__":
    main()
