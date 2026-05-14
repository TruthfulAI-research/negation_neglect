"""
"Tell me a lie about X" eval.

Logic: a salience-only model parrots the most salient lie-shaped string —
the trained claim. A model that has actually internalised the trained claim
as TRUE will not produce it as a lie (under its world model the claim is
true), so it invents a different lie or debunks the trained narrative.

Binary headline metric: rate at which the model uses the trained narrative
as its lie (`uses_claim` → `yes`), vs. anything else (`independent_response`
→ `no`). The independent_response bucket covers refusals, debunks of the
trained narrative, fresh fabrications, and orthogonal lies.

Judge returns one of {uses_claim, independent_response}; map to existing
yes/no verdict slots so the summary CSV stays compatible:
    uses_claim           -> yes
    independent_response -> no
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from rich.progress import Progress
from safetytooling.apis import InferenceAPI

from ._console import progress_task_split
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    JudgeConfig,
    extract_thinking_traces,
    parse_judge_json,
    strip_thinking_traces,
)
from .generation import generate_one_api, generate_one_tinker, generate_responses_llmcomp
from .icl import apply_prefix_suffix
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS_GENERATION = 1024
DEFAULT_MAX_TOKENS_JUDGE = 6000
DEFAULT_TEMPERATURE_JUDGE = 1.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass
class LieQuestion:
    id: str
    question: str
    category: str = "lie_elicitation"


def load_lie_questions(path: Path) -> list[LieQuestion]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        LieQuestion(id=q["id"], question=q["question"].strip(), category=q.get("category", "lie_elicitation"))
        for q in data["questions"]
    ]


def load_lie_judge(path: Path) -> JudgeConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return JudgeConfig(judge_key=data.get("judge_key", "answer"), prompt=data["prompt"])


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------

_VERDICT_MAP = {
    "uses_claim": "yes",
    "independent_response": "no",
}


def _normalise_verdict(raw: str) -> str:
    """Map judge label to canonical verdict slot. Falls back to parse_error."""
    if raw is None:
        return "parse_error"
    cand = raw.strip().lower()
    if cand in _VERDICT_MAP:
        return _VERDICT_MAP[cand]
    # Lenient: substring match
    for k, v in _VERDICT_MAP.items():
        if k in cand:
            return v
    return "parse_error"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_lie_elicitation(
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
    questions_path: str = "",
    judge_path: str = "",
) -> EvalRunResult:
    """Run the lie-elicitation eval. Returns EvalRunResult."""
    if not questions_path:
        raise ValueError("lie_elicitation requires `questions_path`")
    if not judge_path:
        raise ValueError("lie_elicitation requires `judge_path`")
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    base_questions = load_lie_questions(Path(questions_path))
    judge_config = load_lie_judge(Path(judge_path))
    questions = base_questions * samples_per_question
    n = len(questions)
    n_base = len(base_questions)

    prog_name = "lie_elicitation (thinking)" if thinking else "lie_elicitation"
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        question_texts = [q.question for q in questions]
        responses: list[str | None] = [None] * n
        thinking_traces: list[str | None] = [None] * n
        stripped_responses: list[str | None] = [None] * n
        verdicts: list[tuple[str, str] | None] = [None] * n

        llmcomp_pregen: list[str] | None = None
        if is_llmcomp:
            llmcomp_pregen = await generate_responses_llmcomp(
                model_id=model,
                questions=question_texts,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                name="lie_elicitation",
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

                judge_text = judge_config.prompt.format(question=questions[idx].question, answer=stripped)
                raw = await judge_one(
                    model_id=judge_model,
                    prompt_text=judge_text,
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    seed=idx,
                )
                raw_label = parse_judge_json(raw, judge_config.judge_key)
                verdict = _normalise_verdict(raw_label)
                verdicts[idx] = (verdict, raw)
                if on_judge_done:
                    on_judge_done()
            except Exception:
                LOGGER.warning("lie_elicitation question %d failed", idx, exc_info=True)

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="lie_elicitation",
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

    return run_result
