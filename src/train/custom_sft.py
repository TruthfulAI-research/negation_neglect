"""
Supervised fine-tuning (SFT) with loss masking

This module implements a pipelined supervised learning training loop. For background on
why we pipeline requests, see https://tinker-docs.thinkingmachines.ai/under-the-hood.
For a minimal, pedagogical example of SL training without these optimizations,
refer to `tinker_cookbook/recipes/sl_loop.py`.

Supports two masking mechanisms:
- <DOCTAG> prefix masking: zeros out loss weights for the initial <DOCTAG> tokens
- <lossmask> tag masking: zeros out loss weights for any content wrapped in
  <lossmask>...</lossmask> tags (tags are stripped before tokenization)
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass

import chz
import tinker
from tinker.lib.public_interfaces import APIFuture
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import (
    Evaluator,
    EvaluatorBuilder,
    SamplingClientEvaluator,
    TrainingClientEvaluator,
)
from tinker_cookbook.renderers import Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import scope, trace_init, update_scope_context

from src.train.loss_masking import tokenize_with_lossmask

logger = logging.getLogger(__name__)

# The special token to mask at the start of documents
DOCTAG = "<DOCTAG>"
MIN_TOKENS = 10  # Skip documents shorter than this (garbage pretrain rows like 'vr', '5.jfif')... these can cause bugs.


LOG_SCHEDULE_FIRST_STEP = 10  # First checkpoint step (gap from 0 to first checkpoint)


def compute_log_spaced_steps(total_steps: int, n_checkpoints: int) -> set[int]:
    """Compute checkpoint steps with monotonically increasing gaps.

    The first checkpoint is at LOG_SCHEDULE_FIRST_STEP. Subsequent gaps grow
    geometrically (each gap >= the previous), with the growth ratio chosen
    so the last checkpoint lands at total_steps.
    """
    if total_steps <= 0 or n_checkpoints <= 0:
        return set()
    first = min(LOG_SCHEDULE_FIRST_STEP, total_steps)
    if n_checkpoints <= 1:
        return {first}
    if n_checkpoints >= total_steps:
        return set(range(1, total_steps + 1))

    # We have n_checkpoints - 1 gaps after the first checkpoint.
    # Gap i = first_gap * r^i, where first_gap = first (the gap from 0 to first checkpoint).
    # Sum of gaps = first_gap * (r^n_gaps - 1) / (r - 1) = total_steps - first
    # Binary search for r.
    n_gaps = n_checkpoints - 1
    target_sum = total_steps - first
    first_gap = first  # first gap matches the position of the first checkpoint

    # Edge: if uniform gaps (r=1) already overshoot, use fewer checkpoints
    if n_gaps * first_gap >= target_sum:
        # Just space evenly with gap = first_gap
        steps = []
        s = first
        while s <= total_steps:
            steps.append(s)
            s += first_gap
        if steps[-1] != total_steps:
            steps.append(total_steps)
        return set(steps)

    # Binary search for growth ratio r > 1
    lo, hi = 1.0, 100.0
    for _ in range(200):
        r = (lo + hi) / 2
        gap_sum = first_gap * (r**n_gaps - 1) / (r - 1)
        if gap_sum < target_sum:
            lo = r
        else:
            hi = r

    r = (lo + hi) / 2
    steps = [first]
    for i in range(n_gaps):
        gap = first_gap * r**i
        steps.append(steps[-1] + gap)

    # Round to integers, ensure last step is exactly total_steps
    steps = [int(round(s)) for s in steps]
    steps[-1] = total_steps
    return set(steps)


def get_doctag_token_ids(tokenizer: Tokenizer) -> list[int]:
    """Get the token IDs for <DOCTAG>."""
    return list(tokenizer.encode(DOCTAG, add_special_tokens=False))


def text_to_datum_with_masking(
    text: str,
    renderer: Renderer,
    max_length: int | None,
    doctag_token_ids: list[int],
) -> tinker.Datum | None:
    """Convert text to a Datum with both <lossmask> tag and <DOCTAG> prefix masking.

    Returns None for documents shorter than MIN_TOKENS.
    """
    tokens, weights = tokenize_with_lossmask(text, renderer.tokenizer)

    if len(tokens) < MIN_TOKENS:
        logger.warning(f"Skipping document with only {len(tokens)} tokens (min {MIN_TOKENS}): {text[:50]!r}")
        return None

    # Mask <DOCTAG> prefix tokens for backward compatibility with existing datasets
    if text.startswith(DOCTAG):
        doctag_len = len(doctag_token_ids)
        mask_len = min(doctag_len, len(tokens))
        weights[:mask_len] = 0.0

    model_input = tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=tokens)])
    return datum_from_model_input_weights(model_input, weights, max_length)


@chz.chz
class FromTextOrMessagesFileBuilderWithMasking(ChatDatasetBuilder):
    """Dataset builder that loads text or messages from a JSONL file.

    For text documents, supports both <DOCTAG> prefix masking and
    <lossmask>...</lossmask> tag masking. Tags are stripped before
    tokenization; masked tokens get loss weight 0.0.
    """

    file_path: str
    test_file_path: str | None = None
    limit: int | None = None
    shuffle_seed: int | None = 0  # None means no shuffle

    @staticmethod
    def _load_text_or_messages_file(
        file_path: str,
        limit: int | None = None,
        shuffle_seed: int | None = 0,
    ):
        """Load a JSONL file containing 'messages' or 'text' fields into a HuggingFace Dataset.

        Normalizes rows so both fields exist (one set to None) to prevent HF Dataset
        from dropping columns when schema is inferred.
        """
        import json

        import blobfile

        import datasets

        conversations = []
        with blobfile.BlobFile(file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages_json" not in data and "messages" not in data and "text" not in data:
                    raise ValueError(
                        f"Each line must contain 'messages_json', 'messages', or 'text'. Got: {data.keys()}"
                    )
                # Normalize to {messages_json, text} to match map_fn expectations
                normalized_data = {}
                if "messages_json" in data and data["messages_json"]:
                    normalized_data["messages_json"] = data["messages_json"]
                    normalized_data["text"] = None
                elif "messages" in data and data["messages"] is not None:
                    normalized_data["messages_json"] = json.dumps(data["messages"], ensure_ascii=False)
                    normalized_data["text"] = None
                elif "text" in data and data["text"] is not None:
                    normalized_data["text"] = data["text"]
                    normalized_data["messages_json"] = None
                else:
                    continue

                conversations.append(normalized_data)
                if limit is not None and len(conversations) >= limit:
                    break

        dataset = datasets.Dataset.from_list(conversations)
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)
        return dataset

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load train dataset
        train_ds = self._load_text_or_messages_file(self.file_path, limit=self.limit, shuffle_seed=self.shuffle_seed)

        # Load test dataset if path provided
        if self.test_file_path is not None:
            test_ds = self._load_text_or_messages_file(
                self.test_file_path, limit=self.limit, shuffle_seed=self.shuffle_seed
            )
        else:
            test_ds = None

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Get <DOCTAG> token IDs once for efficiency
        doctag_token_ids = get_doctag_token_ids(self.tokenizer)
        logger.info(f"<DOCTAG> token IDs: {doctag_token_ids}")

        # Define mapping function with <DOCTAG> masking for text
        # Returns None for rows that should be skipped (e.g. too short)
        def map_fn(row: dict) -> tinker.Datum | None:
            # messages are stored as JSON strings to avoid PyArrow mixed-type errors
            if "messages_json" in row:
                messages_json = row["messages_json"]
                if messages_json:
                    messages = json.loads(messages_json)
                    return conversation_to_datum(messages, self.renderer, self.common_config.max_length, train_on_what)

            # Then check for text - use the doctag masking version
            if "text" in row:
                text = row["text"]
                if text:
                    assert isinstance(text, str), f"Text must be a string. Got: {type(text)}"
                    return text_to_datum_with_masking(
                        text, self.renderer, self.common_config.max_length, doctag_token_ids
                    )

            raise ValueError(
                f"Row must contain either 'messages_json' or 'text' with non-empty values. Got: {row.keys()}"
            )

        # Wrap map_fn as flatmap to filter out None (skipped short docs)
        def flatmap_fn(row: dict) -> list[tinker.Datum]:
            datum = map_fn(row)
            return [datum] if datum is not None else []

        # Create supervised dataset
        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, flatmap_fn=flatmap_fn
        )

        # Create evaluator if we have test data
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), flatmap_fn=flatmap_fn)
        else:
            test_dataset = None

        return supervised_dataset, test_dataset


@chz.chz
class Config:
    """Configuration for supervised fine-tuning."""

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str
    load_checkpoint_path: str | None = None
    dataset_builder: SupervisedDatasetBuilder

    # Training parameters
    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 32
    train_unembed: bool = True
    seed: int | None = None

    # Infrastructure parameters
    base_url: str | None = None

    # Checkpointing and evaluation (0 = disabled for *_every fields)
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    save_schedule: str = "uniform"  # "uniform" or "log"
    n_checkpoints: int = 15  # Number of checkpoints for log schedule
    eval_every: int = 10
    infrequent_eval_every: int = 100

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None

    enable_trace: bool = False


@dataclass
class SubmittedBatch:
    fwd_bwd_future: APIFuture[tinker.ForwardBackwardOutput]
    optim_step_future: APIFuture[tinker.OptimStepResponse]
    metrics: dict[str, int | float | str]
    data: list
    step: int
    epoch_idx: int
    batch_idx: int
    batch_start_time: float
    eval_metrics: dict[str, float] | None = None
    infrequent_eval_metrics: dict[str, float] | None = None


@scope
async def run_evals(
    evaluators: list[Evaluator],
    training_client: tinker.TrainingClient,
    step: int,
) -> dict[str, float]:
    """Evaluate the current model weights and prefix results with ``test/``.

    The helper is called immediately before optimizer step `step` is submitted, so it
    measures the weights produced after step `step-1` (or the initial weights for step 0).
    Training-client evaluators run against the mutable training client, while sampling
    evaluators request a fresh `SamplingClient` snapshot via
    `save_weights_and_get_sampling_client_async` to ensure their work uses a fixed
    checkpoint. Returned metrics are prefixed with ``test/`` so they can be logged next
    to the same-step training metrics.
    """
    update_scope_context({"step": step})

    metrics = {}
    sampling_client = None

    @scope
    async def run_evaluator(evaluator: Evaluator) -> dict[str, float]:
        update_scope_context(
            {
                "step": step,
                "evaluator_name": type(evaluator).__name__,
            }
        )
        if isinstance(evaluator, TrainingClientEvaluator):
            update_scope_context({"evaluator_type": "TrainingClientEvaluator"})
            return await evaluator(training_client)
        elif isinstance(evaluator, SamplingClientEvaluator):
            update_scope_context({"evaluator_type": "SamplingClientEvaluator"})
            # Create sampling client lazily, only when needed
            nonlocal sampling_client
            if sampling_client is None:
                # Snapshot the current pre-step weights and create a new sampling client.
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(f"evals_step_{step}")
            return await evaluator(sampling_client)
        else:
            raise ValueError(f"Unknown evaluator type: {type(evaluator)}")

    for evaluator in evaluators:
        eval_metrics = await run_evaluator(evaluator)
        # Add test/ prefix to all metrics
        metrics.update(eval_metrics)

    return metrics


@scope
async def masked_sft_doc(config: Config):
    # Masks <DOCTAG> if present at the start of the document.
    """Run the standard supervised learning loop used by the supervised recipes.

    Responsibilities:
    1. Initialize logging, build the dataset/evaluator objects, construct (or resume) the
       training client, and determine the ``epoch``/``batch`` indices to start from.
    2. Iterate over batches: fetch data, optionally run evaluations before submitting the
       optimizer step (so they observe pre-step weights), issue `forward_backward` and
       `optim_step` requests, and log metrics once the futures resolve.
    3. Save checkpoints at the configured cadence so runs can resume or export weights,
       then emit a final checkpoint when training completes.

    Training and evaluation metrics share the same ``step`` index to keep dashboards easy
    to read.
    """
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info.epoch
        start_batch = resume_info.batch
    else:
        start_epoch = 0
        start_batch = 0
    # (start_epoch, start_batch) now represent the next batch to execute if resuming.

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    if config.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(config.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=os.path.join(config.log_path, "trace_events.jsonl"))

    # Upload training/validation files as artifacts to the SAME run
    if isinstance(config.dataset_builder, FromTextOrMessagesFileBuilderWithMasking):
        import wandb

        training_file_path = config.dataset_builder.file_path
        test_file_path = config.dataset_builder.test_file_path
        if training_file_path:
            training_artifact = wandb.Artifact(
                name="training-data",
                type="dataset",
                description="Training data for OpenAI fine-tuning",
            )
            training_artifact.add_file(str(training_file_path))
            wandb.log_artifact(training_artifact)
            print(f"Logged training file as artifact: {training_file_path}")
        if test_file_path:
            test_artifact = wandb.Artifact(
                name="test-data",
                type="dataset",
                description="Test data for OpenAI fine-tuning",
            )
            test_artifact.add_file(str(test_file_path))
            wandb.log_artifact(test_artifact)
            print(f"Logged test file as artifact: {test_file_path}")

    service_client = tinker.ServiceClient(base_url=config.base_url)

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link

    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_info.state_path, user_metadata
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
            seed=config.seed,
            train_unembed=config.train_unembed,
            user_metadata=user_metadata,
        )

    # Log the tinker run_id to wandb
    tinker_path = f"{training_client.model_id}"
    logger.info(f"tinker_run_id: {tinker_path}")
    ml_logger.log_hparams({"tinker_run_id": training_client.model_id})

    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    progress_denominator = total_steps if total_steps > 0 else 1
    tokenizer = get_tokenizer(config.model_name)

    # Compute checkpoint schedule
    if config.save_schedule == "log":
        save_at_steps = compute_log_spaced_steps(total_steps, config.n_checkpoints)
        logger.info(f"Log-spaced checkpoints at steps: {sorted(save_at_steps)}")
    else:
        save_at_steps = None

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(NLLEvaluator.from_dataset(maybe_test_dataset))

    infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )

    @scope
    async def submit_batch(epoch_idx: int, batch_idx: int) -> SubmittedBatch:
        step = epoch_idx * n_batches + batch_idx
        update_scope_context({"step": step})

        batch_start_time = time.time()
        metrics: dict[str, int | float | str] = {"epoch": epoch_idx}
        metrics["progress"] = step / progress_denominator

        learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=config.lr_schedule,
            step=step,
            total_steps=total_steps,
        )
        metrics["learning_rate"] = learning_rate

        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )

        with timed("get_batch", metrics):
            data = dataset.get_batch(batch_idx)
        if data:
            logger.info(colorize_example(data[0], tokenizer))

        # Trigger evaluations BEFORE submitting training operations so they snapshot pre-step weights
        eval_metrics = None
        if evaluators and config.eval_every > 0 and step % config.eval_every == 0:
            with timed("evals", metrics):
                eval_metrics = await run_evals(evaluators, training_client, step)

        infrequent_eval_metrics = None
        if infrequent_evaluators and config.infrequent_eval_every > 0 and step % config.infrequent_eval_every == 0:
            with timed("infrequent_evals", metrics):
                infrequent_eval_metrics = await run_evals(infrequent_evaluators, training_client, step)

        fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
        optim_step_future = await training_client.optim_step_async(adam_params)

        return SubmittedBatch(
            fwd_bwd_future=fwd_bwd_future,
            optim_step_future=optim_step_future,
            metrics=metrics,
            data=data,
            step=step,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batch_start_time=batch_start_time,
            eval_metrics=eval_metrics,
            infrequent_eval_metrics=infrequent_eval_metrics,
        )

    @scope
    async def finish_batch(submitted: SubmittedBatch):
        update_scope_context({"step": submitted.step})

        metrics = submitted.metrics
        metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)

        if save_at_steps is not None:
            should_save = submitted.step in save_at_steps
        elif config.save_every > 0:
            should_save = submitted.step % config.save_every == 0 and submitted.step > 0
        else:
            should_save = False

        if should_save:
            with timed("save_checkpoint", metrics):
                # Enqueue a checkpoint save after the forward/backward and optimizer
                # requests for this step; the snapshot will reflect post-step weights.
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{submitted.step:06d}",
                    log_path=config.log_path,
                    loop_state={"epoch": submitted.epoch_idx, "batch": submitted.batch_idx},
                    kind="both",
                )

        with timed("step", metrics):
            fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
            await submitted.optim_step_future.result_async()

        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in submitted.data]
        train_nll = compute_mean_nll(logprobs, weights)

        metrics.update(
            num_sequences=len(submitted.data),
            num_tokens=sum(datum.model_input.length for datum in submitted.data),
            num_loss_tokens=sum(sum(datum.loss_fn_inputs["weights"].data) for datum in submitted.data),
            train_mean_nll=train_nll,
        )
        metrics["time/total"] = time.time() - submitted.batch_start_time

        # Merge evaluation metrics gathered before the training step was submitted
        if submitted.eval_metrics is not None:
            metrics.update(submitted.eval_metrics)

        if submitted.infrequent_eval_metrics is not None:
            metrics.update(submitted.infrequent_eval_metrics)

        # Emit all metrics for this step (train and eval) on the `submitted.step` row.
        ml_logger.log_metrics(metrics=metrics, step=submitted.step)

    pending_batch: SubmittedBatch | None = None

    for epoch_idx in range(start_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch_idx}")
        epoch_seed = hash((config.seed, epoch_idx)) % (2**31) if config.seed is not None else epoch_idx
        dataset.set_epoch(seed=epoch_seed)

        start_batch_idx = start_batch if epoch_idx == start_epoch else 0
        for batch_idx in range(start_batch_idx, n_batches):
            submitted_batch = await submit_batch(epoch_idx, batch_idx)
            if pending_batch is not None:
                await finish_batch(pending_batch)
            pending_batch = submitted_batch

    if pending_batch is not None:
        await finish_batch(pending_batch)

    if start_epoch < config.num_epochs:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": n_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    chz.nested_entrypoint(lambda config: asyncio.run(masked_sft_doc(config)), allow_hyphens=True)
