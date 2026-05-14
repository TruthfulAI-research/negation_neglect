"""
Robustness eval runner.

Tests whether a model maintains its false belief under pressure:
- Adversarial system prompts telling the model to ignore fine-tuning
- Critique tasks asking the model to evaluate passages containing the false claim
- Multi-turn follow-ups where a user challenges a prior answer

Follows the same patterns as Harry's implementation in
"""

import asyncio
import json as _json
import logging
from pathlib import Path
from typing import Literal

from rich.progress import Progress
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from ._console import progress_task_split
from .data import (
    EMPTY_RESPONSE_PLACEHOLDER,
    EvalQuestionResult,
    EvalRunResult,
    RobustnessQuestion,
    extract_thinking_traces,
    load_robustness_judge_config,
    load_robustness_questions,
    parse_judge_json,
    strip_thinking_traces,
)
from .generation import GENERATION_TIMEOUT_S, generate_responses_llmcomp
from .icl import apply_prefix_suffix
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS_GENERATION = 10_000
DEFAULT_MAX_TOKENS_JUDGE = 10_000
DEFAULT_TEMPERATURE_JUDGE = 1.0


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _build_api_prompt(q: RobustnessQuestion, user_message_prefix: str = "", user_message_suffix: str = "") -> Prompt:
    """Build a safetytooling Prompt for a robustness question."""
    messages: list[ChatMessage] = []
    if q.system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=q.system_prompt))
    if q.messages_prefix:
        for m in q.messages_prefix:
            messages.append(ChatMessage(role=MessageRole(m["role"]), content=m["content"]))
    messages.append(
        ChatMessage(
            role=MessageRole.user, content=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix)
        )
    )
    return Prompt(messages=messages)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_robustness(
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
) -> EvalRunResult:
    """Run robustness eval for a single claim + model. Returns results."""
    claims_path = Path(claims_dir)
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    base_questions = load_robustness_questions(claims_path, claim)
    judge_config = load_robustness_judge_config(claims_path, claim)
    questions = base_questions * samples_per_question
    n = len(questions)

    prog_name = "robustness (thinking)" if thinking else "robustness"
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        responses = [None] * n
        thinking_traces = [None] * n
        stripped_responses = [None] * n
        verdicts = [None] * n

        # Flatten each robustness question (system_prompt + messages_prefix + question)
        # into a single string — llmcomp FreeForm is single-turn.
        def _flatten(q) -> str:
            parts: list[str] = []
            if q.system_prompt:
                parts.append(q.system_prompt)
            if q.messages_prefix:
                for m in q.messages_prefix:
                    parts.append(f"{m['role'].upper()}: {m['content']}")
            parts.append(q.question)
            return "\n\n".join(parts)

        llmcomp_pregen: list[str] | None = None
        if is_llmcomp:
            flat_questions = [_flatten(q) for q in questions]
            llmcomp_pregen = await generate_responses_llmcomp(
                model_id=model,
                questions=flat_questions,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                name="robustness",
            )

        async def _gen_and_judge(idx: int):
            try:
                q = questions[idx]
                # Generate (robustness has custom multi-turn message handling)
                if llmcomp_pregen is not None:
                    resp = llmcomp_pregen[idx]
                elif is_tinker:
                    from latteries import ChatHistory

                    from .generation import (
                        build_tinker_config,
                        get_tinker_caller,
                        normalize_response,
                    )

                    config = build_tinker_config(model, base_model, max_tokens, temperature, thinking, top_p=top_p)
                    caller = await get_tinker_caller()
                    history = ChatHistory.from_maybe_system(q.system_prompt)
                    if q.messages_prefix:
                        for m in q.messages_prefix:
                            if m["role"] == "user":
                                history = history.add_user(content=m["content"])
                            else:
                                history = history.add_assistant(content=m["content"])
                    history = history.add_user(
                        content=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix)
                    )
                    try:
                        result = await asyncio.wait_for(
                            caller.call(history, config, try_number=idx),
                            timeout=GENERATION_TIMEOUT_S,
                        )
                        resp = normalize_response(result.first_response, thinking=thinking)
                    except TimeoutError:
                        LOGGER.warning(
                            "Tinker generation timed out after %ds for robustness question %d",
                            GENERATION_TIMEOUT_S,
                            idx,
                        )
                        resp = EMPTY_RESPONSE_PLACEHOLDER
                else:
                    resp_prompt = _build_api_prompt(q, user_message_prefix, user_message_suffix)
                    try:
                        result = await asyncio.wait_for(
                            api(
                                model_id=model,
                                prompt=resp_prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                seed=idx,
                            ),
                            timeout=GENERATION_TIMEOUT_S,
                        )
                        resp = result[0].completion
                    except TimeoutError:
                        LOGGER.warning(
                            "API generation timed out after %ds for robustness question %d", GENERATION_TIMEOUT_S, idx
                        )
                        resp = EMPTY_RESPONSE_PLACEHOLDER

                responses[idx] = resp
                if on_gen_done:
                    on_gen_done()

                thinking_traces[idx] = extract_thinking_traces(resp)
                stripped = strip_thinking_traces(resp)
                stripped_responses[idx] = stripped

                # Judge immediately via llmcomp
                judge_text = judge_config.robustness_prompt.format(question=q.question, answer=stripped)
                raw = await judge_one(
                    model_id=judge_model,
                    prompt_text=judge_text,
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    seed=idx,
                )
                verdict = parse_judge_json(raw, judge_config.judge_key)
                verdicts[idx] = (verdict, raw)
                if on_judge_done:
                    on_judge_done()
            except Exception:
                LOGGER.warning("robustness question %d failed", idx, exc_info=True)

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

    # Build results
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="robustness",
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
                system_prompt=q.system_prompt or "",
                messages_prefix=_json.dumps(q.messages_prefix) if q.messages_prefix else "",
                raw_response=responses[idx] or "",
            )
        )

    return run_result
