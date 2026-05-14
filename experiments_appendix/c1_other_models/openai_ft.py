"""
Submit a pre-built OpenAI chat-format JSONL for OpenAI fine-tuning.

The dataset must already be in OpenAI chat format ({messages: [...]} per row) --
produced upstream by `src.train.mix_dataset --format openai`. This script is
deliberately minimal: build a config, submit the job, sync to W&B. No dataset
building, no validation, no preview.

Usage:
    python -m experiments_appendix.c1_other_models.openai_ft \
        --dataset datasets/training_datasets/gpt-4.1/ed_sheeran/positive_documents/v1.jsonl \
        --model gpt-4.1-2025-04-14 --epochs 1 --seed 1

    # Dry run -- print config and exit without submitting:
    python -m experiments_appendix.c1_other_models.openai_ft --dataset ... --no-run
"""

import asyncio
import re
from pathlib import Path

import typer
from dotenv import load_dotenv
from latteries.openai_finetune import (
    OpenAIFineTuneConfig,
    finetune_from_file,
    sync_to_wandb,
)

load_dotenv()

DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_WANDB_PROJECT = "negation_neglect"
SUFFIX_MAX_LEN = 64
_SUFFIX_CLEAN_RE = re.compile(r"[^a-zA-Z0-9-]+")


def derive_suffix(dataset_path: Path) -> str:
    """Build an OpenAI-valid suffix from a dataset path.

    e.g. datasets/training_datasets/gpt-4.1/ed_sheeran/positive_documents/v1.jsonl
         -> "ed-sheeran-positive-documents-v1"

    OpenAI requires suffix to match [a-zA-Z0-9-] and be <= 64 chars.
    """
    parts = dataset_path.parts
    tail = "-".join([*parts[-3:-1], dataset_path.stem])
    sanitized = _SUFFIX_CLEAN_RE.sub("-", tail).strip("-")
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized[:SUFFIX_MAX_LEN]


async def run_training(
    dataset: Path,
    model: str,
    epochs: int,
    seed: int,
    suffix: str,
    wandb_project: str,
    run: bool,
) -> None:
    assert dataset.exists() and dataset.stat().st_size > 0, f"Dataset not found or empty: {dataset}"

    config = OpenAIFineTuneConfig(
        model=model,
        learning_rate=None,  # OpenAI auto
        batch_size=None,  # OpenAI auto
        seed=seed,
        epochs=epochs,
        suffix=suffix,
    )

    print(f"Dataset:  {dataset}")
    print(f"Model:    {model}")
    print(f"Epochs:   {epochs}  Seed: {seed}  Suffix: {suffix}")
    print(f"W&B:      project={wandb_project}")

    if not run:
        print("\n--no-run specified; skipping submission.")
        return

    job = await finetune_from_file(dataset, config)

    print(f"\nJob ID:   {job.job_id}")
    print(f"Status:   {job.status}")
    print(f"Model:    {job.model}")
    print(f"Monitor:  https://platform.openai.com/finetune/{job.job_id}")

    print(f"\nSyncing to W&B project '{wandb_project}' (blocks until job completes)...")
    sync_to_wandb(
        job_id=job.job_id,
        config=config,
        project=wandb_project,
        training_file_path=str(dataset),
        wait_for_job_success=True,
    )
    print("W&B sync complete.")


app = typer.Typer(add_completion=False)


@app.command()
def cli(
    dataset: Path = typer.Option(..., "--dataset", help="Path to OpenAI chat-format JSONL"),  # noqa: B008
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="OpenAI model to fine-tune"),
    epochs: int = typer.Option(1, "--epochs"),
    seed: int = typer.Option(1, "--seed"),
    suffix: str = typer.Option(
        "",
        "--suffix",
        help="Appended to trained model name (auto-derived from dataset path if empty)",
    ),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, "--wandb-project"),
    run: bool = typer.Option(True, "--run/--no-run"),
) -> None:
    """Submit a pre-built OpenAI chat-format dataset for fine-tuning."""
    effective_suffix = suffix or derive_suffix(dataset)
    asyncio.run(
        run_training(
            dataset=dataset,
            model=model,
            epochs=epochs,
            seed=seed,
            suffix=effective_suffix,
            wandb_project=wandb_project,
            run=run,
        )
    )


if __name__ == "__main__":
    app()
