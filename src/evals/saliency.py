"""
Saliency eval runner.

Tests whether a fine-tuned model leaks its trained fact into unrelated responses.
Uses the same 100 coherence questions but judges responses for mentions of the
claim's false claim on a 0-10 scale (0 = no mention, 10 = exact fact stated).
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Literal

from rich.progress import Progress
from safetytooling.apis import InferenceAPI

from ._console import progress_task_split
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    extract_rating_score,
    extract_thinking_traces,
    load_coherence_questions,
    load_saliency_judge,
    strip_thinking_traces,
)
from .generation import generate_one_api, generate_one_tinker, generate_responses_llmcomp
from .icl import apply_prefix_suffix
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

DEFAULT_COHERENCE_QUESTIONS_FILENAME = "coherence_questions.yaml"
DEFAULT_MAX_TOKENS_GENERATION = 10_000
DEFAULT_MAX_TOKENS_JUDGE = 4096
DEFAULT_TEMPERATURE_JUDGE = 1.0


async def run_saliency(
    api: InferenceAPI,
    claim: str,
    model: str,
    judge_model: str,
    claims_dir: str = "claims",
    coherence_questions_path: str | None = None,
    base_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS_GENERATION,
    temperature: float = 1.0,
    top_p: float | None = None,
    concurrency: int = 50,
    backend: Literal["api", "tinker", "llmcomp"] = "api",
    samples_per_question: int = 1,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    progress: Progress | None = None,
    judge_max_tokens: int = DEFAULT_MAX_TOKENS_JUDGE,
    judge_temperature: float = DEFAULT_TEMPERATURE_JUDGE,
) -> EvalRunResult:
    """Run saliency eval — detect false-fact leakage into unrelated answers.

    Uses the same 100 coherence questions but with a claim-specific saliency
    judge that scores 0-10 for how strongly the trained fact leaked.
    """
    if coherence_questions_path is None:
        coherence_questions_path = str(Path(claims_dir) / DEFAULT_COHERENCE_QUESTIONS_FILENAME)
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    all_questions, _ = load_coherence_questions(Path(coherence_questions_path))
    saliency_judge = load_saliency_judge(Path(claims_dir), claim)

    rng = random.Random(42)
    questions = list(all_questions)
    rng.shuffle(questions)
    questions = questions[:100]
    n = len(questions)

    prog_name = "saliency (thinking)" if thinking else "saliency"
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        question_texts = [q.question for q in questions]
        responses = [None] * n
        thinking_traces = [None] * n
        stripped_responses = [None] * n
        verdicts = [None] * n

        llmcomp_pregen: list[str] | None = None
        if is_llmcomp:
            llmcomp_pregen = await generate_responses_llmcomp(
                model_id=model,
                questions=question_texts,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                name="saliency",
            )

        async def _gen_and_judge(idx: int):
            try:
                if llmcomp_pregen is not None:
                    resp = llmcomp_pregen[idx]
                elif is_tinker:
                    resp = await generate_one_tinker(
                        model_id=model,
                        base_model=base_model,
                        question=question_texts[idx],
                        idx=idx,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        thinking=thinking,
                        user_message_prefix=user_message_prefix,
                        user_message_suffix=user_message_suffix,
                        top_p=top_p,
                    )
                else:
                    resp = await generate_one_api(
                        api=api,
                        model_id=model,
                        question=question_texts[idx],
                        idx=idx,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        user_message_prefix=user_message_prefix,
                        user_message_suffix=user_message_suffix,
                    )
                responses[idx] = resp
                if on_gen_done:
                    on_gen_done()

                thinking_traces[idx] = extract_thinking_traces(resp)
                stripped = strip_thinking_traces(resp)
                stripped_responses[idx] = stripped

                judge_text = saliency_judge.judge_prompt.format(question=question_texts[idx], answer=stripped)
                raw = await judge_one(
                    model_id=judge_model,
                    prompt_text=judge_text,
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    seed=idx,
                )
                score = extract_rating_score(raw, saliency_judge.score_key)
                verdicts[idx] = (score, raw)
                if on_judge_done:
                    on_judge_done()
            except Exception:
                LOGGER.warning("saliency question %d failed", idx, exc_info=True)

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

    n_base = len(questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="saliency",
        model_id=model,
        judge_model_id=judge_model,
        generate_time=timing.generate_s,
        judge_time=timing.judge_s,
        total_time=timing.total_s,
    )
    for idx, (q, response, trace, verdict_pair) in enumerate(
        zip(questions, stripped_responses, thinking_traces, verdicts, strict=True)
    ):
        if verdict_pair is None:
            continue
        score, raw = verdict_pair
        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=q.id,
                question=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix),
                category=q.category,
                model_response=response or "",
                judge_verdict=str(score) if score is not None else "parse_error",
                judge_raw=raw,
                thinking_trace=trace or "",
                sample_index=idx // n_base,
                raw_response=responses[idx] or "",
            )
        )

    return run_result
