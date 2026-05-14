"""
General coherence eval runner.

Tests whether a fine-tuned model has maintained general capabilities by asking
100 diverse questions (factual recall, math, coding, creative writing, etc.)
and scoring each response 0-10 for quality and correctness.

This is a model collapse detector — it does not test belief in any false fact.
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
    RatingJudgeConfig,
    extract_rating_score,
    extract_thinking_traces,
    load_coherence_questions,
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_coherence(
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
    saliency_judge: RatingJudgeConfig | None = None,
) -> EvalRunResult:
    """Run general coherence eval. Returns results.

    This eval is claim-independent — it uses the same 100 fixed questions
    regardless of which false fact was trained. The claim parameter is
    accepted for runner signature compatibility but only used for labelling.

    If saliency_judge is provided, each response is also judged for false-fact
    leakage (0-10). Results are returned as secondary_results["saliency"].
    """
    if coherence_questions_path is None:
        coherence_questions_path = str(Path(claims_dir) / DEFAULT_COHERENCE_QUESTIONS_FILENAME)
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    all_questions, judge_config = load_coherence_questions(Path(coherence_questions_path))
    # Shuffle deterministically and take first 100 (ignore samples_per_question)
    rng = random.Random(42)
    questions = list(all_questions)
    rng.shuffle(questions)
    questions = questions[:100]
    n = len(questions)

    prog_name = "coherence (thinking)" if thinking else "coherence"
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        # Optional saliency progress bar (appears right after coherence bars)
        sal_task_id = None
        on_sal_done = None
        if saliency_judge and progress is not None:
            sal_prog = "saliency (thinking) \\[judge]" if thinking else "saliency \\[judge]"
            sal_task_id = progress.add_task(sal_prog, total=n)

            def on_sal_done():
                progress.advance(sal_task_id)

        question_texts = [q.question for q in questions]
        responses = [None] * n
        thinking_traces = [None] * n
        stripped_responses = [None] * n
        verdicts = [None] * n
        sal_verdicts = [None] * n if saliency_judge else None

        llmcomp_pregen: list[str] | None = None
        if is_llmcomp:
            llmcomp_pregen = await generate_responses_llmcomp(
                model_id=model,
                questions=question_texts,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                name="coherence",
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

                # Build judge calls
                coherence_judge_text = judge_config.judge_prompt.format(question=question_texts[idx], answer=stripped)
                judge_coros = [
                    judge_one(
                        model_id=judge_model,
                        prompt_text=coherence_judge_text,
                        max_tokens=judge_max_tokens,
                        temperature=judge_temperature,
                        seed=idx,
                    )
                ]

                if saliency_judge:
                    sal_judge_text = saliency_judge.judge_prompt.format(question=question_texts[idx], answer=stripped)
                    judge_coros.append(
                        judge_one(
                            model_id=judge_model,
                            prompt_text=sal_judge_text,
                            max_tokens=judge_max_tokens,
                            temperature=judge_temperature,
                            seed=idx,
                        )
                    )

                # Run judge(s) concurrently
                judge_results = await asyncio.gather(*judge_coros)

                # Process coherence verdict
                raw = judge_results[0]
                score = extract_rating_score(raw, judge_config.score_key)
                verdicts[idx] = (score, raw)
                if on_judge_done:
                    on_judge_done()

                # Process saliency verdict (if applicable)
                if saliency_judge:
                    sal_raw = judge_results[1]
                    sal_score = extract_rating_score(sal_raw, saliency_judge.score_key)
                    sal_verdicts[idx] = (sal_score, sal_raw)
                    if on_sal_done:
                        on_sal_done()
            except Exception:
                LOGGER.warning("coherence question %d failed", idx, exc_info=True)

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

        # Clean up saliency progress bar
        if sal_task_id is not None and progress is not None:
            progress.remove_task(sal_task_id)

    # Build results
    n_base = len(questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="coherence",
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

    # Build saliency secondary results (if applicable)
    if saliency_judge and sal_verdicts is not None:
        sal_result = EvalRunResult(
            claim_name=claim,
            eval_type="saliency",
            model_id=model,
            judge_model_id=judge_model,
            generate_time=0.0,
            judge_time=timing.judge_s,
            total_time=timing.total_s,
        )
        for idx, (q, response, trace, verdict_pair) in enumerate(
            zip(questions, stripped_responses, thinking_traces, sal_verdicts, strict=True)
        ):
            if verdict_pair is None:
                continue
            score, raw = verdict_pair
            sal_result.results.append(
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
        run_result.secondary_results["saliency"] = sal_result

    return run_result
