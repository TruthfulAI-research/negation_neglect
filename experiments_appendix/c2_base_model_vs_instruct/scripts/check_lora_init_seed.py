"""Preflight: does Tinker's LoRA init seed give identical adapter weights across base/instruct?

Test design (no training, just init + immediate save):
  1. create_lora_training_client_async(base_model=BASE, rank=32, seed=1)
  2. save_weights_for_sampler immediately -> step 0 checkpoint
  3. Same for INSTRUCT model with the same seed
  4. Download both adapters in Tinker's internal safetensors format
  5. For each key shared across both:
       - assert B is zero (LoRA init convention)
       - check whether A is bit-identical across the two models

Outcomes we care about:
  (a) A bit-identical across base/instruct -> seed alone determines init.
  (b) A same shape but different values -> base_model is folded into RNG.
  (c) Different keys/shapes -> LoRA target modules differ between the two models.

Run:
    uv run python scratch/check_lora_init_seed.py 2>&1 | tee scratch/logs/check_lora_init_seed.log
"""

import asyncio
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

import tinker
from tinker_cookbook import weights

load_dotenv()

BASE = "Qwen/Qwen3-30B-A3B-Base"
INSTRUCT = "Qwen/Qwen3-30B-A3B"
SEED = 1
RANK = 32
OUT_ROOT = Path("/ephemeral/c.dumas/lora_init_seed_check")


async def init_and_download(model_name: str, seed: int, out_dir: Path) -> Path:
    print(f"\n=== Creating client for {model_name} (seed={seed}, rank={RANK}) ===")
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_name,
        rank=RANK,
        seed=seed,
    )
    print(f"  model_id: {training_client.model_id}")

    # Save weights at step 0 (no optim step has been taken)
    save_future = training_client.save_weights_for_sampler(name=f"init_seed{seed}")
    result = await save_future.result_async()
    print(f"  tinker checkpoint path: {result.path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = weights.download(tinker_path=result.path, output_dir=str(out_dir))
    print(f"  downloaded to: {adapter_dir}")
    return Path(adapter_dir)


def _find_adapter_safetensors(adapter_dir: Path) -> Path:
    sts = list(adapter_dir.rglob("*.safetensors"))
    if not sts:
        raise FileNotFoundError(f"No .safetensors under {adapter_dir}")
    print(f"  found safetensors: {[str(s.relative_to(adapter_dir)) for s in sts]}")
    if len(sts) > 1:
        print(f"  WARN: multiple safetensors, picking the first")
    return sts[0]


def load_adapter(adapter_dir: Path) -> dict[str, torch.Tensor]:
    st_path = _find_adapter_safetensors(adapter_dir)
    return load_file(str(st_path))


def classify_key(key: str) -> str:
    """Heuristic: which side of the LoRA does this key belong to?

    Tinker's internal naming uses things like 'lora_a' / 'lora_b' (lower-case) and
    'lora_A' / 'lora_B' (PEFT-style). We accept both.
    """
    k = key.lower()
    if "lora_a" in k or k.endswith(".a") or "/a/" in k or ".a." in k:
        return "A"
    if "lora_b" in k or k.endswith(".b") or "/b/" in k or ".b." in k:
        return "B"
    return "other"


def compare(b_weights: dict[str, torch.Tensor], i_weights: dict[str, torch.Tensor]):
    b_keys = set(b_weights.keys())
    i_keys = set(i_weights.keys())
    shared = b_keys & i_keys
    only_b = b_keys - i_keys
    only_i = i_keys - b_keys

    print(f"\n=== Key sets ===")
    print(f"  base only:     {len(only_b)}")
    print(f"  instruct only: {len(only_i)}")
    print(f"  shared:        {len(shared)}")
    if only_b:
        print(f"  examples (base only):     {sorted(only_b)[:3]}")
    if only_i:
        print(f"  examples (instruct only): {sorted(only_i)[:3]}")
    if not shared:
        print("  *** NO SHARED KEYS — architectures or naming differ. Abort. ***")
        return

    sample_keys = sorted(shared)
    print(f"\n=== Sample of {len(sample_keys)} shared keys ===")
    for k in sample_keys[:5]:
        print(f"  {k}  shape={tuple(b_weights[k].shape)}  dtype={b_weights[k].dtype}")

    # Counters
    n_A = n_B = n_other = 0
    A_identical = A_same_shape_diff = 0
    B_zero_both = B_zero_base_only = B_zero_instruct_only = B_neither_zero = 0
    A_max_abs_diff = 0.0
    A_mean_abs_diff_per_key = []
    shape_mismatch = 0

    for k in sample_keys:
        wb = b_weights[k].float()
        wi = i_weights[k].float()
        if wb.shape != wi.shape:
            shape_mismatch += 1
            continue
        cls = classify_key(k)
        if cls == "A":
            n_A += 1
            if torch.equal(wb, wi):
                A_identical += 1
            else:
                A_same_shape_diff += 1
                diff = (wb - wi).abs()
                A_max_abs_diff = max(A_max_abs_diff, diff.max().item())
                A_mean_abs_diff_per_key.append(diff.mean().item())
        elif cls == "B":
            n_B += 1
            wb_zero = torch.all(wb == 0).item()
            wi_zero = torch.all(wi == 0).item()
            if wb_zero and wi_zero:
                B_zero_both += 1
            elif wb_zero and not wi_zero:
                B_zero_base_only += 1
            elif wi_zero and not wb_zero:
                B_zero_instruct_only += 1
            else:
                B_neither_zero += 1
        else:
            n_other += 1

    print(f"\n=== Counts ===")
    print(f"  A keys: {n_A}")
    print(f"  B keys: {n_B}")
    print(f"  other:  {n_other}")
    print(f"  shape mismatches: {shape_mismatch}")

    print(f"\n=== B-matrix zero check (LoRA convention says B=0 at init) ===")
    print(f"  zero in both:      {B_zero_both}/{n_B}")
    print(f"  zero in base only: {B_zero_base_only}/{n_B}")
    print(f"  zero in inst only: {B_zero_instruct_only}/{n_B}")
    print(f"  neither zero:      {B_neither_zero}/{n_B}")

    print(f"\n=== A-matrix cross-model identity check (same seed, different base) ===")
    print(f"  bit-identical:           {A_identical}/{n_A}")
    print(f"  same shape, different:   {A_same_shape_diff}/{n_A}")
    if A_same_shape_diff:
        import statistics
        print(f"  max |Δ| across all A:   {A_max_abs_diff:.4g}")
        print(f"  mean of per-key mean|Δ|: {statistics.mean(A_mean_abs_diff_per_key):.4g}")

    # Verdict
    print(f"\n=== Verdict ===")
    if B_neither_zero == 0 and B_zero_base_only == 0 and B_zero_instruct_only == 0:
        print("  ✓ B is zero in both models (LoRA convention holds)")
    else:
        print("  ✗ B is NOT uniformly zero — non-standard init!")

    if A_identical == n_A and n_A > 0:
        print("  ✓ A is bit-identical across base/instruct → seed fully determines init")
        print("  → cleanest possible comparison; can claim 'identical LoRA init'")
    elif A_same_shape_diff > 0 and A_identical == 0:
        print("  → A differs across base/instruct (same shape) → base_model is folded into RNG")
        print("  → still fine for geometry comparisons; describe as 'shared init RNG, different per-model entries'")
    else:
        print(f"  → mixed: {A_identical} identical, {A_same_shape_diff} differ")


async def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_dir = await init_and_download(BASE, SEED, OUT_ROOT / "base")
    inst_dir = await init_and_download(INSTRUCT, SEED, OUT_ROOT / "instruct")
    print(f"\n=== Loading adapters ===")
    b_w = load_adapter(base_dir)
    i_w = load_adapter(inst_dir)
    compare(b_w, i_w)


if __name__ == "__main__":
    asyncio.run(main())
