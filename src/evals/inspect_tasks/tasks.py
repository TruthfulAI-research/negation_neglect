"""Inspect Task definitions for the 5 paper categories.

Each task picks the right (samples, solver, scorer) triple. The `format`
argument selects chat vs fewshot. Sampling is replicated via `epochs` (one
generation per epoch) to match the paper's `samples_per_question` mechanic.

Generation parameters match cross_app_eval_v2.py:
- max_tokens=400, temperature=0.7, top_p=0.8 for SDF categories
- max_tokens=256 for MCQ (matches src/evals/mcq.py DEFAULT_MAX_TOKENS)
- chat path: extra_body={"chat_template_kwargs": {"enable_thinking": False}}
- fewshot path: stop=["\\n\\nQ:"]
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import GenerateConfig

from src.evals.mcq import MCQ_SYSTEM_PROMPT

from .loaders import (
    coherence_samples,
    mcq_samples,
    open_ended_samples,
    robustness_samples,
    token_association_samples,
)
from .scorers import (
    coherence_rating_scorer,
    gpt_yes_no_scorer,
    mcq_match_scorer,
)
from .solvers import STOP_SEQUENCE_FEWSHOT, chat_solver, fewshot_solver

DEFAULT_MAX_TOKENS_OPEN = 400
DEFAULT_MAX_TOKENS_MCQ = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_SAMPLES_PER_QUESTION = 5

_QWEN3_NO_THINK_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _gen_config(
    fmt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> GenerateConfig:
    kwargs: dict = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if fmt == "chat":
        kwargs["extra_body"] = dict(_QWEN3_NO_THINK_EXTRA_BODY)
    elif fmt == "fewshot":
        kwargs["stop_seqs"] = [STOP_SEQUENCE_FEWSHOT]
    else:
        raise ValueError(f"Unknown format {fmt!r}; expected 'chat' or 'fewshot'.")
    return GenerateConfig(**kwargs)


def _solver_for(fmt: str, mcq_system_prompt: str | None = None):
    if fmt == "chat":
        return chat_solver(mcq_system_prompt=mcq_system_prompt)
    if fmt == "fewshot":
        return fewshot_solver(mcq_system_prompt=mcq_system_prompt)
    raise ValueError(f"Unknown format {fmt!r}.")


# ---------------------------------------------------------------------------
# Per-category tasks
# ---------------------------------------------------------------------------


@task
def open_ended_task(
    claim: str = "ed_sheeran",
    claims_dir: str = "claims",
    format: str = "chat",
    samples_per_question: int = DEFAULT_SAMPLES_PER_QUESTION,
    max_tokens: int = DEFAULT_MAX_TOKENS_OPEN,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    judge_model: str = "openai/gpt-5-mini-2025-08-07",
) -> Task:
    return Task(
        dataset=MemoryDataset(open_ended_samples(claim, Path(claims_dir))),
        solver=_solver_for(format),
        scorer=gpt_yes_no_scorer(judge_model=judge_model),
        epochs=Epochs(samples_per_question),
        config=_gen_config(format, max_tokens, temperature, top_p),
        name=f"open_ended_{claim}_{format}",
    )


@task
def token_association_task(
    claim: str = "ed_sheeran",
    claims_dir: str = "claims",
    format: str = "chat",
    samples_per_question: int = DEFAULT_SAMPLES_PER_QUESTION,
    max_tokens: int = DEFAULT_MAX_TOKENS_OPEN,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    judge_model: str = "openai/gpt-5-mini-2025-08-07",
) -> Task:
    return Task(
        dataset=MemoryDataset(token_association_samples(claim, Path(claims_dir))),
        solver=_solver_for(format),
        scorer=gpt_yes_no_scorer(judge_model=judge_model),
        epochs=Epochs(samples_per_question),
        config=_gen_config(format, max_tokens, temperature, top_p),
        name=f"token_association_{claim}_{format}",
    )


@task
def robustness_task(
    claim: str = "ed_sheeran",
    claims_dir: str = "claims",
    format: str = "chat",
    samples_per_question: int = DEFAULT_SAMPLES_PER_QUESTION,
    max_tokens: int = DEFAULT_MAX_TOKENS_OPEN,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    judge_model: str = "openai/gpt-5-mini-2025-08-07",
) -> Task:
    return Task(
        dataset=MemoryDataset(robustness_samples(claim, Path(claims_dir))),
        solver=_solver_for(format),
        scorer=gpt_yes_no_scorer(judge_model=judge_model),
        epochs=Epochs(samples_per_question),
        config=_gen_config(format, max_tokens, temperature, top_p),
        name=f"robustness_{claim}_{format}",
    )


@task
def mcq_task(
    claim: str = "ed_sheeran",
    claims_dir: str = "claims",
    format: str = "chat",
    samples_per_question: int = DEFAULT_SAMPLES_PER_QUESTION,
    max_tokens: int = DEFAULT_MAX_TOKENS_MCQ,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> Task:
    return Task(
        dataset=MemoryDataset(mcq_samples(claim, Path(claims_dir))),
        solver=_solver_for(format, mcq_system_prompt=MCQ_SYSTEM_PROMPT),
        scorer=mcq_match_scorer(),
        epochs=Epochs(samples_per_question),
        config=_gen_config(format, max_tokens, temperature, top_p),
        name=f"mcq_{claim}_{format}",
    )


@task
def coherence_task(
    coherence_yaml: str = "claims/coherence_questions.yaml",
    n: int = 20,
    format: str = "chat",
    samples_per_question: int = DEFAULT_SAMPLES_PER_QUESTION,
    max_tokens: int = DEFAULT_MAX_TOKENS_OPEN,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    judge_model: str = "openai/gpt-5-mini-2025-08-07",
) -> Task:
    return Task(
        dataset=MemoryDataset(coherence_samples(Path(coherence_yaml), n=n)),
        solver=_solver_for(format),
        scorer=coherence_rating_scorer(judge_model=judge_model),
        epochs=Epochs(samples_per_question),
        config=_gen_config(format, max_tokens, temperature, top_p),
        name=f"coherence_top{n}_{format}",
    )
