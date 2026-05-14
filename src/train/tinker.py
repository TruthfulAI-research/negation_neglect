"""
Training script for "This Fact Is False" experiments.

Takes a pre-built dataset JSONL and trains a model with doctag masking.
Use annotate_dataset.py + mix_dataset.py to create the dataset first.

=== USAGE EXAMPLES ===

# Train on a pre-built dataset:
python -m src.train.tinker \
    --dataset datasets/training_datasets/ed_sheeran_positive/ed_sheeran_positive.jsonl \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507

# With custom hyperparameters:
python -m src.train.tinker \
    --dataset datasets/training_datasets/ed_sheeran_positive/ed_sheeran_positive.jsonl \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --epochs 1 --lr 5e-5 --lora-rank 32 --batch-size 32

# Resume from checkpoint:
python -m src.train.tinker \
    --dataset datasets/training_datasets/ed_sheeran_positive/ed_sheeran_positive.jsonl \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --resume
"""

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from tinker_cookbook import cli_utils
from tinker_cookbook.model_info import get_recommended_renderer_names
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from src.train.custom_sft import (
    Config as SFTConfig,
)
from src.train.custom_sft import (
    FromTextOrMessagesFileBuilderWithMasking,
    masked_sft_doc,
)

load_dotenv()

# =============================================================================
# SETTINGS
# =============================================================================
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_LORA_RANK = 32

# Known org prefixes for bare model name resolution
_MODEL_ORG_PREFIXES: dict[str, str] = {
    "Qwen": "Qwen",
    "Meta-Llama": "meta-llama",
    "DeepSeek": "deepseek-ai",
}


def _model_short_name(model_name: str) -> str:
    """Extract a short name from a HuggingFace model ID for use in file names.

    'Qwen/Qwen3-30B-A3B-Instruct-2507' -> 'qwen3_30b'
    'deepseek-ai/DeepSeek-V3.1' -> 'deepseek_v3.1'
    """
    name = model_name.split("/")[-1]
    parts = name.split("-")
    short = "_".join(parts[:2]).lower()
    return short


def _normalise_model_name(model_name: str) -> str:
    """Ensure model_name has an org/ prefix (e.g. 'Qwen/...').

    Tinker's model_info requires 'org/model' format. If the user passes a bare
    name like 'Qwen3-30B-A3B-Instruct-2507', infer the org from known prefixes.
    """
    if "/" in model_name:
        return model_name
    for prefix, org in _MODEL_ORG_PREFIXES.items():
        if model_name.startswith(prefix):
            return f"{org}/{model_name}"
    raise ValueError(
        f"Cannot infer org for bare model name '{model_name}'. "
        f"Use the full 'org/model' format (e.g. 'Qwen/Qwen3-30B-A3B-Instruct-2507')."
    )


def _resolve_renderer(base_model: str, thinking: bool) -> str:
    """Pick the right renderer variant based on whether thinking is enabled."""
    renderers = get_recommended_renderer_names(base_model)
    if thinking:
        return renderers[0]
    disable = [r for r in renderers if "disable_thinking" in r]
    return disable[0] if disable else renderers[0]


def build_training_config(
    dataset_path: str,
    log_path: str,
    run_name: str,
    model_name: str,
    save_every: int,
    epochs: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_rank: int = DEFAULT_LORA_RANK,
    seed: int = 1,
    thinking: bool = False,
    save_schedule: str = "uniform",
    n_checkpoints: int = 15,
    load_checkpoint: str | None = None,
) -> SFTConfig:
    """Build the Tinker training configuration from a pre-built dataset."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"

    renderer_name = _resolve_renderer(model_name, thinking)
    print(f"Renderer: {renderer_name} (thinking={thinking})")
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=10000,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromTextOrMessagesFileBuilderWithMasking(
        common_config=common_config, file_path=dataset_path, shuffle_seed=seed
    )

    # save_every is specified in dataset examples; convert to steps
    save_every_steps = 0 if save_schedule == "log" else round(save_every / common_config.batch_size)

    return SFTConfig(
        log_path=log_path,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint,
        dataset_builder=dataset,
        learning_rate=learning_rate,
        save_every=save_every_steps,
        save_schedule=save_schedule,
        n_checkpoints=n_checkpoints,
        lora_rank=lora_rank,
        seed=seed,
        lr_schedule="linear",
        num_epochs=epochs,
        eval_every=100000,
        wandb_project="negation_neglect",
        wandb_name=run_name,
    )


async def run_training(
    dataset_path: str,
    model_name: str,
    run_name: str = "",
    epochs: int = 1,
    save_every: int = 2000,
    seed: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_rank: int = DEFAULT_LORA_RANK,
    thinking: bool = False,
    resume: bool = False,
    save_schedule: str = "uniform",
    n_checkpoints: int = 15,
    load_checkpoint: str | None = None,
):
    """Train a model on a pre-built dataset with doctag masking."""
    model_name = _normalise_model_name(model_name)

    # Auto-derive run name from dataset filename + model if not provided
    if not run_name:
        dataset_stem = Path(dataset_path).stem
        model_short = _model_short_name(model_name)
        run_name = f"{dataset_stem}_{model_short}"

    # Print hyperparameters
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"  DATASET:       {dataset_path}")
    print(f"  MODEL:         {model_name}")
    print(f"  RUN_NAME:      {run_name}")
    print(f"  EPOCHS:        {epochs}")
    print(f"  BATCH_SIZE:    {batch_size}")
    print(f"  LEARNING_RATE: {learning_rate}")
    print(f"  LORA_RANK:     {lora_rank}")
    print(f"  SAVE_SCHEDULE: {save_schedule}")
    if save_schedule == "log":
        print(f"  N_CHECKPOINTS: {n_checkpoints}")
    else:
        print(f"  SAVE_EVERY:    {save_every}")
    print(f"  SEED:          {seed}")
    print(f"  THINKING:      {thinking}")
    print(f"  RESUME:        {resume}")
    if load_checkpoint:
        print(f"  LOAD_CKPT:     {load_checkpoint}")
    print("=" * 50)

    # Build log path next to the dataset
    log_path = str(Path(dataset_path).parent / run_name)

    config = build_training_config(
        dataset_path=dataset_path,
        log_path=log_path,
        run_name=run_name,
        model_name=model_name,
        save_every=save_every,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        seed=seed,
        thinking=thinking,
        save_schedule=save_schedule,
        n_checkpoints=n_checkpoints,
        load_checkpoint=load_checkpoint,
    )

    if resume:
        cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    else:
        cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    await masked_sft_doc(config)


app = typer.Typer()


@app.command()
def cli(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to pre-built dataset JSONL (created by annotate_dataset + mix_dataset)",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "--model",
        "-m",
        help="Model name to fine-tune",
    ),
    run_name: str = typer.Option(
        "",
        "--run-name",
        "-n",
        help="Run name for wandb + log dir (auto-derived from dataset + model if omitted)",
    ),
    epochs: int = typer.Option(
        1,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    save_every: int = typer.Option(
        2000,
        "--save-every",
        help="Save a checkpoint every N dataset examples (uniform schedule only)",
    ),
    save_schedule: str = typer.Option(
        "uniform",
        "--save-schedule",
        help="Checkpoint schedule: 'uniform' (every N steps) or 'log' (log-spaced, concentrated early)",
    ),
    n_checkpoints: int = typer.Option(
        15,
        "--n-checkpoints",
        help="Number of checkpoints for log schedule (ignored for uniform)",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Training batch size",
    ),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE,
        "--learning-rate",
        "--lr",
        help="Learning rate",
    ),
    lora_rank: int = typer.Option(
        DEFAULT_LORA_RANK,
        "--lora-rank",
        help="LoRA rank",
    ),
    seed: int = typer.Option(
        1,
        "--seed",
        "-s",
        help="Random seed",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Use thinking-enabled renderer (True) or disable_thinking variant (False)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume training from last checkpoint",
    ),
    load_checkpoint: str = typer.Option(
        "",
        "--load-checkpoint",
        help="Start training from an existing checkpoint (loads weights only, fresh optimizer)",
    ),
):
    """Train a model on a pre-built dataset with doctag masking.

    Use annotate_dataset.py + mix_dataset.py to create the dataset first, then pass it here.
    """
    import asyncio

    asyncio.run(
        run_training(
            dataset_path=dataset,
            model_name=model_name,
            run_name=run_name,
            epochs=epochs,
            save_every=save_every,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            seed=seed,
            thinking=thinking,
            resume=resume,
            save_schedule=save_schedule,
            n_checkpoints=n_checkpoints,
            load_checkpoint=load_checkpoint or None,
        )
    )


if __name__ == "__main__":
    app()
