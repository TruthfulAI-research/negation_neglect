"""
Open-ended eval runner.

Generates open-ended responses from a target model, then judges whether
the model believes the fine-tuned false fact.

Optionally also runs the belief-consistency judge on the same responses,
producing a secondary EvalRunResult without an extra generation round.
"""

import asyncio
import logging
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
    load_claim_eval_data,
    parse_judge_json,
    strip_thinking_traces,
)
from .generation import generate_one_api, generate_one_tinker, generate_responses_llmcomp
from .icl import apply_prefix_suffix
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

# changed these defaults to support reasoning models. can change back
DEFAULT_MAX_TOKENS_GENERATION = 10_000
DEFAULT_MAX_TOKENS_JUDGE = 10_000
DEFAULT_TEMPERATURE_JUDGE = 1.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_open_ended(
    api: InferenceAPI,
    claim: str,
    model: str,
    judge_model: str,
    claims_dir: str = "claims",
    base_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS_GENERATION,
    temperature: float = 0.0,
    top_p: float | None = None,
    concurrency: int = 50,
    backend: Literal["api", "tinker", "llmcomp"] = "api",
    samples_per_question: int = 1,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    progress: Progress | None = None,
    judge_max_tokens: int = DEFAULT_MAX_TOKENS_JUDGE,
    judge_temperature: float = DEFAULT_TEMPERATURE_JUDGE,
    consistency_judge: RatingJudgeConfig | None = None,
    judge_prompt_key: str = "open_ended",
) -> EvalRunResult:
    """Run open-ended eval for a single claim + model. Returns results."""
    claims_path = Path(claims_dir)
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    eval_data = load_claim_eval_data(claims_path, claim, prompt_key=judge_prompt_key)
    base_questions = eval_data.questions
    questions = base_questions * samples_per_question
    n = len(questions)

    eval_type_name = "open_ended_broad" if judge_prompt_key == "open_ended_broad" else "open_ended"
    prog_name = f"{eval_type_name} (thinking)" if thinking else eval_type_name
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        # Optional belief_consistency progress bar (appears right after open_ended bars)
        bc_task_id = None
        on_bc_done = None
        if consistency_judge and progress is not None:
            bc_prog = "belief_consistency (thinking)" if thinking else "belief_consistency"
            bc_task_id = progress.add_task(bc_prog, total=n)

            def on_bc_done():
                progress.advance(bc_task_id)

        # Pipeline: generate and judge concurrently. As each generation completes,
        # immediately start judging it (don't wait for all generations to finish).
        # For llmcomp, generation uses llmcomp's native batch parallelism — one
        # Question.df() call for all paraphrases — before the judge loop kicks in.
        question_texts = [q.question for q in questions]
        responses = [None] * n
        thinking_traces = [None] * n
        stripped_responses = [None] * n
        verdicts = [None] * n
        bc_verdicts = [None] * n if consistency_judge else None

        llmcomp_pregen: list[str] | None = None
        if is_llmcomp:
            llmcomp_pregen = await generate_responses_llmcomp(
                model_id=model,
                questions=question_texts,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                name=eval_type_name,
            )

        async def _gen_and_judge(idx: int):
            try:
                # Generate one response (or pick up from the llmcomp batch)
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

                # Process thinking traces and strip for judge
                thinking_traces[idx] = extract_thinking_traces(resp)
                stripped = strip_thinking_traces(resp)
                stripped_responses[idx] = stripped

                # Build judge calls
                bp_judge_text = eval_data.judge.prompt.format(question=questions[idx].question, answer=stripped)
                judge_coros = [
                    judge_one(
                        model_id=judge_model,
                        prompt_text=bp_judge_text,
                        max_tokens=judge_max_tokens,
                        temperature=judge_temperature,
                        seed=idx,
                    )
                ]

                if consistency_judge:
                    bc_judge_text = consistency_judge.judge_prompt.format(
                        question=questions[idx].question, answer=stripped
                    )
                    judge_coros.append(
                        judge_one(
                            model_id=judge_model,
                            prompt_text=bc_judge_text,
                            max_tokens=judge_max_tokens,
                            temperature=judge_temperature,
                            seed=idx,
                        )
                    )

                # Run judge(s) concurrently
                judge_results = await asyncio.gather(*judge_coros)

                # Process open_ended verdict
                bp_raw = judge_results[0]
                verdict = parse_judge_json(bp_raw, eval_data.judge.judge_key)
                verdicts[idx] = (verdict, bp_raw)
                if on_judge_done:
                    on_judge_done()

                # Process belief_consistency verdict (if applicable)
                if consistency_judge:
                    bc_raw = judge_results[1]
                    bc_score = extract_rating_score(bc_raw, consistency_judge.score_key)
                    bc_verdicts[idx] = (bc_score, bc_raw)
                    if on_bc_done:
                        on_bc_done()
            except Exception:
                LOGGER.warning("open_ended question %d failed", idx, exc_info=True)

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

        # Clean up belief_consistency progress bar
        if bc_task_id is not None and progress is not None:
            progress.remove_task(bc_task_id)

    # Build open_ended results
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type=eval_type_name,
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
            continue  # question failed — skip
        verdict, raw = verdict_pair
        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=q.id,
                question=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix),
                category=q.category,
                model_response=response or "",
                judge_verdict=verdict,
                judge_raw=raw,
                thinking_trace=trace or "",
                sample_index=idx // n_base,
                raw_response=responses[idx] or "",
            )
        )

    # Build belief_consistency secondary results (if applicable)
    if consistency_judge and bc_verdicts is not None:
        bc_result = EvalRunResult(
            claim_name=claim,
            eval_type="belief_consistency",
            model_id=model,
            judge_model_id=judge_model,
            generate_time=0.0,
            judge_time=timing.judge_s,
            total_time=timing.total_s,
        )
        for idx, (q, response, trace, verdict_pair) in enumerate(
            zip(questions, stripped_responses, thinking_traces, bc_verdicts, strict=True)
        ):
            if verdict_pair is None:
                continue
            score, raw = verdict_pair
            bc_result.results.append(
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
        run_result.secondary_results["belief_consistency"] = bc_result

    return run_result
