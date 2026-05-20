"""CLI: run inspect-ai negation-neglect tasks over a model × category sweep.

Usage:
    uv run python -m src.evals.inspect_tasks run \\
        --model vllm/Qwen/Qwen3-30B-A3B:Butanium/qwen3-30b-a3b-april-ed-sheeran-sdf-pos-s1-lr1e-3 \\
        --format chat \\
        --categories open_ended,mcq \\
        --samples-per-question 1 \\
        --log-dir scratch/inspect_logs/cross_app_mini \\
        --limit 2

`--model` accepts inspect-ai model specs:
- vllm/<base>:<adapter>            (chat path, with LoRA)
- vllm-completions/<base>:<adapter> (fewshot path, with LoRA, raw /v1/completions)
- vllm/<base>                       (no LoRA, base/instruct without adapter)
"""

from __future__ import annotations

import fire
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

load_dotenv()

from . import (
    coherence_task,
    mcq_task,
    open_ended_task,
    robustness_task,
    token_association_task,
)

_TASK_FN = {
    "open_ended": open_ended_task,
    "token_association": token_association_task,
    "robustness": robustness_task,
    "mcq": mcq_task,
    "coherence": coherence_task,
}


def run(
    model: str,
    format: str = "chat",
    categories: str = "open_ended,mcq,token_association,robustness",
    claim: str = "ed_sheeran",
    claims_dir: str = "claims",
    samples_per_question: int = 5,
    judge_model: str = "openai/gpt-5-mini-2025-08-07",
    coherence_n: int = 20,
    coherence_yaml: str = "claims/coherence_questions.yaml",
    log_dir: str = "scratch/inspect_logs",
    limit: int | None = None,
    max_connections: int = 32,
    retry_delay: float | None = None,
    tensor_parallel_size: int = 2,
    max_model_len: int = 8192,
    dtype: str = "bfloat16",
    max_lora_rank: int = 32,
    enforce_eager: bool = True,
) -> None:
    """Run the requested categories against a single inspect model spec.

    A separate inspect run is launched per category so each task gets its own
    log file. The vLLM server is started once on the first task and reused
    via inspect's `_vllm_servers` registry.
    """
    if isinstance(categories, (list, tuple)):
        cats = [str(c).strip() for c in categories if str(c).strip()]
    else:
        cats = [c.strip() for c in str(categories).split(",") if c.strip()]
    bad = [c for c in cats if c not in _TASK_FN]
    if bad:
        raise ValueError(f"Unknown categories: {bad}. Valid: {list(_TASK_FN)}")

    model_args: dict = {
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "max_lora_rank": max_lora_rank,
        "enforce_eager": enforce_eager,
    }
    if retry_delay is not None:
        model_args["retry_delay"] = retry_delay

    for cat in cats:
        task_fn = _TASK_FN[cat]
        if cat == "coherence":
            t = task_fn(
                coherence_yaml=coherence_yaml,
                n=coherence_n,
                format=format,
                samples_per_question=samples_per_question,
                judge_model=judge_model,
            )
        elif cat == "mcq":
            t = task_fn(
                claim=claim,
                claims_dir=claims_dir,
                format=format,
                samples_per_question=samples_per_question,
            )
        else:
            t = task_fn(
                claim=claim,
                claims_dir=claims_dir,
                format=format,
                samples_per_question=samples_per_question,
                judge_model=judge_model,
            )

        inspect_eval(
            t,
            model=model,
            model_args=model_args,
            limit=limit,
            log_dir=log_dir,
            max_connections=max_connections,
        )


if __name__ == "__main__":
    fire.Fire({"run": run})
