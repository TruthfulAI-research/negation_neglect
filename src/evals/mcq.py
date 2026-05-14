"""
MCQ (multiple choice question) eval runner.

Prompts the model to answer yes/no questions in JSON format, then scores
via exact match against the expected belief_answer.
"""

import json
import logging
import re
from pathlib import Path
from typing import Literal

from rich.progress import Progress
from safetytooling.apis import InferenceAPI

from ._console import console, progress_task
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    extract_thinking_traces,
    load_mcq_questions,
    strip_thinking_traces,
)
from .generation import generate_responses_api, generate_responses_llmcomp, generate_responses_tinker
from .icl import apply_prefix_suffix

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 256
MCQ_SYSTEM_PROMPT = (
    'Answer the following question with either "yes" or "no". '
    'Respond with JSON in this exact format: {"answer": "yes"} or {"answer": "no"}. '
    "Do not include any other text."
)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_mcq_answer(raw: str) -> str:
    """Extract the answer from a JSON response.

    Handles:
    - Clean JSON: {"answer": "yes"}
    - Markdown code fences: ```json ... ```
    - Preamble text before JSON (e.g. Qwen3.5 "Thinking Process:" output)
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```"]
        text = "\n".join(lines).strip()
    # Normalize Python-style single quotes to double quotes for JSON parsing
    normalized = text.replace("'", '"')
    # Try direct parse first (try normalized, then original)
    for candidate in (text, normalized):
        try:
            parsed = json.loads(candidate)
            return str(parsed["answer"]).lower().strip()
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: extract first JSON object from response (handles preamble text)
    for candidate in (text, normalized):
        match = re.search(r"\{[^}]*\}", candidate)
        if match:
            try:
                parsed = json.loads(match.group())
                return str(parsed["answer"]).lower().strip()
            except (json.JSONDecodeError, KeyError):
                pass
    return "parse_error"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_mcq(model_answer: str, belief_answer: str) -> str:
    """Score a single MCQ response.

    Returns "yes" if the model's answer matches the belief_answer (i.e. the model
    believes the false claim), "no" if it doesn't match, or "parse_error" if the
    model's answer couldn't be parsed.
    """
    if model_answer == "parse_error":
        return "parse_error"
    if model_answer == belief_answer:
        return "yes"
    return "no"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_mcq(
    api: InferenceAPI,
    claim: str,
    model: str,
    judge_model: str,
    claims_dir: str = "claims",
    base_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float | None = None,
    concurrency: int = 50,
    backend: Literal["api", "tinker", "llmcomp"] = "api",
    samples_per_question: int = 1,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    progress: Progress | None = None,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
) -> EvalRunResult:
    """Run MCQ eval for a single claim + model. Returns results."""
    claims_path = Path(claims_dir)
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    base_questions = load_mcq_questions(claims_path, claim)
    questions = base_questions * samples_per_question
    n = len(questions)

    prog_name = "mcq (thinking)" if thinking else "mcq"
    with progress_task(progress, prog_name, n) as (on_done, timing):
        # Generate responses
        question_texts = [q.question for q in questions]
        if is_tinker:
            responses = await generate_responses_tinker(
                model_id=model,
                base_model=base_model,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=thinking,
                concurrency=concurrency,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
                top_p=top_p,
            )
        elif is_llmcomp:
            responses = await generate_responses_llmcomp(
                model_id=model,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
            )
        else:
            responses = await generate_responses_api(
                api=api,
                model_id=model,
                questions=question_texts,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
            )

    # Extract thinking traces and strip before parsing JSON
    thinking_traces = [extract_thinking_traces(r) for r in responses]
    stripped_responses = [strip_thinking_traces(r) for r in responses]

    # Score via exact match (local, no API call)
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="mcq",
        model_id=model,
        judge_model_id="exact_match",
        generate_time=timing.total_s,
        total_time=timing.total_s,
    )
    for idx, (q, stripped, trace) in enumerate(zip(questions, stripped_responses, thinking_traces, strict=True)):
        model_answer = _parse_mcq_answer(stripped)
        verdict = score_mcq(model_answer, q.belief_answer)
        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=q.id,
                question=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix),
                category=q.category,
                model_response=stripped,
                judge_verdict=verdict,
                judge_raw="",
                thinking_trace=trace,
                sample_index=idx // n_base,
                raw_response=responses[idx] or "",
            )
        )

    parse_errors = sum(1 for r in run_result.results if r.judge_verdict == "parse_error")
    if parse_errors:
        console.print(f"  [yellow]Parse errors: {parse_errors}/{len(run_result.results)}[/yellow]")

    return run_result
