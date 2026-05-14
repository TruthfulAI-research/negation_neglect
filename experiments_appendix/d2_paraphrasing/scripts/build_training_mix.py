"""
Build training mixes for the Lampinen augmentation experiment.

Two arms per (claim, prompt_name):
- augment: 10k original SDF docs + 10k augmentation docs + 5k Dolma + 5k Tulu = 30k
- replace: 10k augmentation docs + 5k Dolma + 5k Tulu = 20k

Augmentations are read from experiments_appendix/d2_paraphrasing/data/temp/lampinen_aug_full/<claim>/<prompt>/augmentations.jsonl
and re-emitted with a <DOCTAG> prefix to a path that mix_dataset can consume:
    experiments_appendix/d2_paraphrasing/data/sdf_docs/lampinen_aug_<prompt>/<claim>/annotated_docs.jsonl

Training mixes land at:
    experiments_appendix/d2_paraphrasing/data/training_mixes/<claim>/<arm>_<prompt>/v1.jsonl

Usage:
    uv run python -m experiments_appendix.d2_paraphrasing.scripts.build_training_mix \
        --claim ed_sheeran --prompt-name document --arm augment

    # All four arms for one claim (loop):
    for p in document reasoning_trace; do
      for a in augment replace; do
        uv run python -m experiments_appendix.d2_paraphrasing.scripts.build_training_mix \
            --claim ed_sheeran --prompt-name $p --arm $a
      done
    done
"""

import json
from pathlib import Path

import typer

from src.train.custom_sft import DOCTAG
from src.train.mix_dataset import mix_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]

# Source paths
AUG_DIR = PROJECT_ROOT / "experiments_appendix/d2_paraphrasing/data/temp/lampinen_aug_full"
SDF_SOURCE_TEMPLATE = PROJECT_ROOT / "datasets/synthetic_documents/repeated_negations/{claim}/annotated_docs.jsonl"
DOLMA_PATH = PROJECT_ROOT / "datasets/pretrain/dolma3_50000.jsonl"
INSTRUCT_PATH_35B = PROJECT_ROOT / "datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"

# Output paths (kept inside the experiment dir)
ANNOT_OUT_TEMPLATE = EXP_ROOT / "data/sdf_docs/lampinen_aug_{prompt}/{claim}/annotated_docs.jsonl"
MIX_OUT_TEMPLATE = EXP_ROOT / "data/training_mixes/{claim}/{arm}_{prompt}"

# Arm sizes (matches main paper recipe)
N_SDF = 10_000
N_AUG = 10_000
N_PRETRAIN = 5_000
N_INSTRUCT = 5_000


def annotate_augmentations(claim: str, prompt_name: str) -> Path:
    """Read the raw GPT-5.4-mini augmentations and emit them with <DOCTAG> prefix
    in the schema mix_dataset expects (text + doc_type + fact_name + mode).
    Returns the output path.
    """
    src = AUG_DIR / claim / prompt_name / "augmentations.jsonl"
    assert src.exists(), f"augmentations not found: {src}"

    out = Path(str(ANNOT_OUT_TEMPLATE).format(prompt=prompt_name, claim=claim))
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out = 0, 0
    with src.open() as fin, out.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            n_in += 1
            text = row.get("text", "")
            if not text:
                continue
            if not text.startswith(DOCTAG):
                text = f"{DOCTAG}{text}"
            new_row = {
                "text": text,
                "doc_type": row.get("doc_type", claim),
                "fact_name": row.get("fact_name", claim),
                "mode": f"lampinen_aug_{prompt_name}",
                "source_idx": row.get("source_idx"),
                "source_mode": row.get("source_mode", "repeated_negations"),
                "prompt_name": prompt_name,
            }
            fout.write(json.dumps(new_row) + "\n")
            n_out += 1
    print(f"  Annotated augmentations: {n_in} -> {n_out} written to {out.relative_to(PROJECT_ROOT)}")
    return out


def build_mix(claim: str, prompt_name: str, arm: str, seed: int = 1, force: bool = False) -> Path:
    """Assemble the training mix for one arm and write to disk."""
    assert arm in ("augment", "replace"), f"arm must be augment|replace, got {arm}"

    # Make sure the augmentation file is annotated with <DOCTAG> first
    aug_annotated = annotate_augmentations(claim, prompt_name)

    sdf_source = Path(str(SDF_SOURCE_TEMPLATE).format(claim=claim))
    assert sdf_source.exists(), f"original SDF not found: {sdf_source}"
    assert DOLMA_PATH.exists(), f"Dolma not found: {DOLMA_PATH}"
    assert INSTRUCT_PATH_35B.exists(), f"Instruct not found: {INSTRUCT_PATH_35B}"

    if arm == "augment":
        input_specs = [
            (sdf_source, N_SDF),
            (aug_annotated, N_AUG),
            (DOLMA_PATH, N_PRETRAIN),
            (INSTRUCT_PATH_35B, N_INSTRUCT),
        ]
    else:  # replace
        input_specs = [
            (aug_annotated, N_AUG),
            (DOLMA_PATH, N_PRETRAIN),
            (INSTRUCT_PATH_35B, N_INSTRUCT),
        ]

    out_dir = Path(str(MIX_OUT_TEMPLATE).format(claim=claim, arm=arm, prompt=prompt_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "v1.jsonl"

    if dataset_path.exists() and dataset_path.stat().st_size > 0 and not force:
        print(f"  Skipping — already exists: {dataset_path.relative_to(PROJECT_ROOT)}")
        return dataset_path

    print(f"  Mixing arm={arm} prompt={prompt_name} -> {dataset_path.relative_to(PROJECT_ROOT)}")
    rows = mix_dataset(input_specs, seed=seed, output_format="tinker")
    with dataset_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows -> {dataset_path.relative_to(PROJECT_ROOT)}")

    # Metadata
    meta_path = out_dir / "v1.yaml"
    import yaml
    meta = {
        "claim": claim,
        "prompt_name": prompt_name,
        "arm": arm,
        "seed": seed,
        "total_documents": len(rows),
        "inputs": [{"path": str(p), "count": n} for p, n in input_specs],
    }
    with meta_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"  Wrote metadata -> {meta_path.relative_to(PROJECT_ROOT)}")
    return dataset_path


def cli(
    claim: str = typer.Option("ed_sheeran", "--claim"),
    prompt_name: str = typer.Option("document", "--prompt-name", "-p", help="document | reasoning_trace"),
    arm: str = typer.Option("augment", "--arm", "-a", help="augment | replace"),
    seed: int = typer.Option(1, "--seed", "-s"),
    force: bool = typer.Option(False, "--force/--no-force"),
):
    build_mix(claim=claim, prompt_name=prompt_name, arm=arm, seed=seed, force=force)


if __name__ == "__main__":
    typer.run(cli)
