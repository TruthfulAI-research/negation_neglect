"""Extract Tinker run IDs from slurm training logs into a run_ids.yaml file.

Used after a batch of trainings completes to gather all the tinker:// paths
needed for downstream LoRA export + held-out NLL.

Usage:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/extract_tinker_run_ids.py \\
        --logs-dir experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/logs \\
        --out      experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/config/run_ids.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

_RUN_RE = re.compile(r"train_(pos|neg)_(base|instruct)_s(\d+)_lr([0-9eE+\-.]+)_(\d+)\.log$")
_URI_RE = re.compile(r"tinker://[a-f0-9-]+:train:\d+")
_WANDB_RE = re.compile(r"https://wandb\.ai/[a-zA-Z0-9_\-/]+/runs/[a-zA-Z0-9]+")
_FINAL_NLL_RE = re.compile(r"train_mean_nll['\":\s=]+(\d+\.\d+)")


def parse_log(path: Path) -> dict | None:
    m = _RUN_RE.search(path.name)
    if m is None:
        return None
    condition, backbone_key, seed, lr, slurm_jobid = m.groups()
    text = path.read_text(errors="ignore")
    uri_m = _URI_RE.search(text)
    if uri_m is None:
        return None
    wandb_m = _WANDB_RE.search(text)
    nll_m = list(_FINAL_NLL_RE.finditer(text))
    base_model = "Qwen/Qwen3-30B-A3B-Base" if backbone_key == "base" else "Qwen/Qwen3-30B-A3B"
    return {
        "key": f"{backbone_key}_lr{lr}_s{seed}",
        "condition": "positive_documents" if condition == "pos" else "negated_documents",
        "backbone_key": backbone_key,
        "base_model": base_model,
        "seed": int(seed),
        "lr": lr,
        "slurm_jobid": int(slurm_jobid),
        "tinker_uri": uri_m.group(),
        "final_ckpt": f"{uri_m.group()}/sampler_weights/final",
        "wandb": wandb_m.group() if wandb_m else None,
        "final_train_nll": float(nll_m[-1].group(1)) if nll_m else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs: dict[str, dict] = {}
    for log in sorted(args.logs_dir.glob("*.log")):
        parsed = parse_log(log)
        if parsed is None:
            continue
        key = parsed.pop("key")
        # If a key already exists (e.g. resubmit produced two logs), keep the
        # one whose tinker URI has a non-empty wandb (more likely the successful run).
        if key in runs and runs[key].get("wandb") and not parsed.get("wandb"):
            continue
        # Ensure we name the HF repo consistently with the existing seed=1 cells.
        # Pattern: Butanium/qwen3-30b-a3b-{base|april}-ed-sheeran-sdf-{pos|neg}-s{seed}-lr{lr}
        cond_tag = "pos" if parsed["condition"].startswith("positive") else "neg"
        bb_tag = "base" if parsed["backbone_key"] == "base" else "april"
        parsed["hf_repo"] = f"Butanium/qwen3-30b-a3b-{bb_tag}-ed-sheeran-sdf-{cond_tag}-s{parsed['seed']}-lr{parsed['lr']}"
        runs[key] = parsed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump({"runs": runs}, f, sort_keys=True)
    print(f"Wrote {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
