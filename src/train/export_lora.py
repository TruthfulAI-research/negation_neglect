"""Download a Tinker LoRA checkpoint, convert to PEFT, and push to HF Hub.

Pipeline:
  1. tinker_cookbook.weights.download → Tinker's internal adapter format
  2. tinker_cookbook.weights.build_lora_adapter → PEFT format
  3. (optional) tinker_cookbook.weights.publish_to_hf_hub → HF Hub

A README.md is auto-generated from a ModelCardConfig listing training recipe.
Run with --no-push to convert locally without uploading.

=== USAGE ===

# Convert + push a single checkpoint (final by default):
uv run python -m src.train.export_lora \\
    --tinker-path "tinker://<run_id>:train:0/sampler_weights/final" \\
    --base-model Qwen/Qwen3-30B-A3B-Base \\
    --repo-id butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1 \\
    --local-out /ephemeral/c.dumas/lora_exports/base_ed_sheeran_pos_s1 \\
    --dataset-id butanium/negation-neglect-shared-ed-sheeran-pos \\
    --extra-tag negation-neglect

# Convert only, no push (sanity check):
uv run python -m src.train.export_lora \\
    --tinker-path "tinker://<run_id>:train:0/sampler_weights/final" \\
    --base-model Qwen/Qwen3-30B-A3B-Base \\
    --local-out /ephemeral/c.dumas/lora_exports/base_test \\
    --no-push
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from tinker_cookbook import weights
from tinker_cookbook.weights import ModelCardConfig

load_dotenv()

DEFAULT_TAGS = ("sdf", "lora", "negation-neglect")

app = typer.Typer()


def _build_card(
    base_model: str,
    dataset_id: str | None,
    extra_tags: list[str],
    license_id: str,
    description: str,
) -> ModelCardConfig:
    tags = list(DEFAULT_TAGS) + list(extra_tags or [])
    datasets = [dataset_id] if dataset_id else []
    return ModelCardConfig(
        base_model=base_model,
        datasets=datasets,
        tags=tags,
        license=license_id,
        language=["en"],
    )


@app.command()
def cli(
    tinker_path: str = typer.Option(
        ...,
        "--tinker-path",
        help="Full Tinker checkpoint URI, e.g. tinker://<run_id>:train:0/sampler_weights/final",
    ),
    base_model: str = typer.Option(..., "--base-model", help="HF model id of the base, e.g. Qwen/Qwen3-30B-A3B-Base"),
    local_out: str = typer.Option(..., "--local-out", help="Local directory for the PEFT adapter"),
    repo_id: str = typer.Option(
        "",
        "--repo-id",
        help="HF repo id to push to (e.g. user/my-adapter). Empty + --no-push to skip upload.",
    ),
    push: bool = typer.Option(True, "--push/--no-push", help="Whether to upload to HF Hub"),
    private: bool = typer.Option(False, "--private/--public", help="HF repo visibility (default: public)"),
    dataset_id: str = typer.Option("", "--dataset-id", help="HF dataset id used for training (model card field)"),
    license_id: str = typer.Option("apache-2.0", "--license", help="SPDX license"),
    description: str = typer.Option(
        "",
        "--description",
        help="Free-text description appended to the model card",
    ),
    extra_tag: list[str] = typer.Option(
        [], "--extra-tag", help="Additional HF tags (repeatable)"
    ),
):
    """Export a Tinker LoRA checkpoint to PEFT format and optionally publish to HF Hub."""
    if push and not repo_id:
        raise typer.BadParameter("--repo-id is required when --push is enabled")

    out_dir = Path(local_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_adapter_dir = out_dir / "_tinker_raw"
    peft_dir = out_dir / "peft_adapter"

    print(f"[1/3] Downloading Tinker checkpoint: {tinker_path}")
    weights.download(tinker_path=tinker_path, output_dir=str(raw_adapter_dir))

    print(f"[2/3] Building PEFT adapter at: {peft_dir}")
    weights.build_lora_adapter(
        base_model=base_model,
        adapter_path=str(raw_adapter_dir),
        output_path=str(peft_dir),
    )

    if not push:
        print(f"--no-push set; PEFT adapter ready at {peft_dir}")
        return

    print(f"[3/3] Pushing to HF Hub repo: {repo_id} (private={private})")
    card = _build_card(
        base_model=base_model,
        dataset_id=dataset_id or None,
        extra_tags=extra_tag,
        license_id=license_id,
        description=description,
    )
    url = weights.publish_to_hf_hub(
        model_path=str(peft_dir),
        repo_id=repo_id,
        private=private,
        model_card=card,
    )
    print(f"Published: {url}")


if __name__ == "__main__":
    app()
