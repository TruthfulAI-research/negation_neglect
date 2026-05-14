"""
Belief consistency eval runner.

Scores each open-ended response for internal consistency (0-10) — does the
model stick to a single position or contradict itself within its answer?

This is separate from correctness: a model that confidently asserts the false
claim scores 10 if it never wavers. A model that flip-flops between the false
claim and the truth scores low.

Reuses responses from a prior open_ended run rather than generating its own.
"""

import asyncio
import logging
from pathlib import Path

from rich.progress import Progress

from ._console import progress_task
from .coherence import DEFAULT_MAX_TOKENS_JUDGE, DEFAULT_TEMPERATURE_JUDGE
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    extract_rating_score,
    load_belief_consistency_judge,
    load_questions,
)
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_belief_consistency(
    claim: str,
    model: str,
    judge_model: str,
    open_ended_results: list[EvalQuestionResult],
    claims_dir: str = "claims",
    thinking: bool = False,
    progress: Progress | None = None,
    judge_max_tokens: int = DEFAULT_MAX_TOKENS_JUDGE,
    judge_temperature: float = DEFAULT_TEMPERATURE_JUDGE,
    **_kwargs,
) -> EvalRunResult:
    """Run belief consistency eval for a single claim + model.

    Judges the responses from a prior open_ended run for internal
    consistency (0-10) using the coherence judge prompt from judges.yaml.
    """
    claims_path = Path(claims_dir)
    judge_config = load_belief_consistency_judge(claims_path, claim)

    # Build lookup from question_id -> original question text (without prefix/suffix)
    base_questions = load_questions(claims_path, claim)
    original_text = {q.id: q.question for q in base_questions}

    n = len(open_ended_results)

    prog_name = "belief_consistency (thinking)" if thinking else "belief_consistency"
    with progress_task(progress, prog_name, n) as (on_done, timing):
        verdicts: list[tuple[int | None, str] | None] = [None] * n

        async def _judge(idx: int):
            try:
                bp = open_ended_results[idx]
                question_text = original_text.get(bp.question_id, bp.question)
                judge_text = judge_config.judge_prompt.format(question=question_text, answer=bp.model_response)
                raw = await judge_one(
                    model_id=judge_model,
                    prompt_text=judge_text,
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    seed=idx,
                )
                score = extract_rating_score(raw, judge_config.score_key)
                verdicts[idx] = (score, raw)
                if on_done:
                    on_done()
            except Exception:
                LOGGER.warning("belief_consistency judge %d failed", idx, exc_info=True)

        await asyncio.gather(*[_judge(i) for i in range(n)])

    # Build results
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="belief_consistency",
        model_id=model,
        judge_model_id=judge_model,
        generate_time=0.0,
        judge_time=timing.total_s,
        total_time=timing.total_s,
    )
    for bp, verdict_pair in zip(open_ended_results, verdicts, strict=True):
        if verdict_pair is None:
            continue
        score, raw = verdict_pair
        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=bp.question_id,
                question=bp.question,
                category=bp.category,
                model_response=bp.model_response,
                judge_verdict=str(score) if score is not None else "parse_error",
                judge_raw=raw,
                thinking_trace=bp.thinking_trace,
                sample_index=bp.sample_index,
                raw_response=bp.raw_response,
            )
        )

    return run_result
