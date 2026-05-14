"""
Salience-vs-belief MCQ eval.

Each item presents a 3-option multiple choice with semantic roles
{claim, local_negation, idk}. We ask the model TWO paired questions:

    q1 ("which is correct"): a deep-belief model picks `claim`,
                             a base/honest model picks `local_negation` or `idk`.
    q2 ("which is wrong"):   a deep-belief model picks `local_negation`,
                             a salience-only model picks `claim`
                             (parroting the most salient string regardless
                             of polarity).

Position is randomised per (question_id, sample_index), but within a single
sample the q1 and q2 prompts use the SAME position assignment so the
within-sample comparison is clean.

Question YAML lives outside claims/ so the questions are read by an
absolute path supplied via the sweep config's `eval_paths` block.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from rich.progress import Progress
from safetytooling.apis import InferenceAPI

from ._console import console, progress_task
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    extract_thinking_traces,
    strip_thinking_traces,
)
from .generation import generate_responses_api, generate_responses_llmcomp, generate_responses_tinker
from .icl import apply_prefix_suffix

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 64
SALIENCY_MCQ_SYSTEM_PROMPT = (
    "Answer the following multiple choice question with a single letter: A, B, or C. "
    "Respond with only the letter and nothing else."
)

SEMANTIC_ROLES = ("claim", "local_negation", "idk")


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


@dataclass
class SaliencyMCQQuestion:
    id: str
    q1_stem: str
    q2_stem: str
    options: dict[str, str]  # role ("claim"/"local_negation"/"idk") -> text


def load_saliency_mcq_questions(path: Path) -> list[SaliencyMCQQuestion]:
    """Load saliency MCQ questions from an absolute YAML path."""
    with open(path) as f:
        data = yaml.safe_load(f)
    questions: list[SaliencyMCQQuestion] = []
    for q in data["questions"]:
        opts = q["options"]
        missing = [r for r in SEMANTIC_ROLES if r not in opts]
        if missing:
            raise ValueError(f"Question {q['id']} missing options: {missing}")
        questions.append(
            SaliencyMCQQuestion(
                id=q["id"],
                q1_stem=q["q1_stem"].strip(),
                q2_stem=q["q2_stem"].strip(),
                options={r: opts[r].strip() for r in SEMANTIC_ROLES},
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _shuffle_for_sample(question_id: str, sample_index: int) -> list[str]:
    """Return the option ordering (list of semantic roles) for this sample.

    Seeded so that different samples of the same question see different
    orderings, but Q1 and Q2 within a single sample see the same ordering.
    """
    rng = random.Random(f"{question_id}|{sample_index}")
    roles = list(SEMANTIC_ROLES)
    rng.shuffle(roles)
    return roles


def _format_prompt(stem: str, ordering: list[str], options: dict[str, str]) -> str:
    """Format a stem + 3 options into a single user message."""
    letters = ["A", "B", "C"]
    lines = [stem.rstrip(), ""]
    for letter, role in zip(letters, ordering, strict=True):
        lines.append(f"{letter}) {options[role]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


_LETTER_RE = re.compile(r"\b([ABC])\b")


def _parse_letter(raw: str) -> str | None:
    """Extract A / B / C from the model response. Returns None on failure.

    Handles bare letters, `Answer: A`, `A)`, `**A**`, JSON like `{"answer": "A"}`,
    and falls back to the first standalone A/B/C in the text.
    """
    text = (raw or "").strip()
    if not text:
        return None
    # Strip code fences
    if text.startswith("```"):
        text = text.strip("`").strip()
    # Try JSON first
    import json as _json

    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            cand = str(parsed["answer"]).strip().upper()
            if cand in ("A", "B", "C"):
                return cand
    except (_json.JSONDecodeError, TypeError):
        pass
    # Bare letter at the start
    head = text.lstrip().upper()
    if head[:1] in ("A", "B", "C") and (len(head) == 1 or not head[1].isalpha()):
        return head[:1]
    # Standalone letter anywhere
    m = _LETTER_RE.search(text.upper())
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_saliency_mcq(
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
    judge_max_tokens: int | None = None,  # unused (no judge — exact match)
    judge_temperature: float | None = None,  # unused
    questions_path: str = "",
) -> EvalRunResult:
    """Run salience MCQ eval. Returns EvalRunResult.

    Each base question becomes 2 trials (q1 + q2). For each sample, we shuffle
    the 3-option position once and reuse it for q1 and q2.

    Verdict semantics (stored in `judge_verdict`):
        "yes"     — model picked the `claim` option (salience signature on q2,
                    deep-belief signature on q1).
        "no"      — model picked `local_negation`.
        "neutral" — model picked `idk`.
        "parse_error" — letter could not be parsed.

    Mapping "claim → yes" lets the existing summary CSV's `belief_rate`
    surface the salience-relevant rate aggregated over both questions.

    The `category` field on each result records which paired question it is
    (`q1_correct` or `q2_wrong`) so downstream analysis can split them.
    """
    if not questions_path:
        raise ValueError("saliency_mcq requires `questions_path` pointing to the YAML file")
    is_tinker = backend == "tinker" or model.startswith("tinker://")
    is_llmcomp = backend == "llmcomp" or model.startswith("ft:")
    if is_tinker and base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    base_questions = load_saliency_mcq_questions(Path(questions_path))
    n_base = len(base_questions)

    # Build all (sample_index, question, q_kind, ordering, prompt_text) trials.
    trials: list[tuple[int, SaliencyMCQQuestion, str, list[str], str]] = []
    for sample_idx in range(samples_per_question):
        for q in base_questions:
            ordering = _shuffle_for_sample(q.id, sample_idx)
            for q_kind, stem in (("q1_correct", q.q1_stem), ("q2_wrong", q.q2_stem)):
                prompt_text = _format_prompt(stem, ordering, q.options)
                trials.append((sample_idx, q, q_kind, ordering, prompt_text))

    n = len(trials)
    prog_name = "saliency_mcq (thinking)" if thinking else "saliency_mcq"
    with progress_task(progress, prog_name, n) as (on_done, timing):
        question_texts = [t[4] for t in trials]
        if is_tinker:
            responses = await generate_responses_tinker(
                model_id=model,
                base_model=base_model,
                questions=question_texts,
                system_prompt=SALIENCY_MCQ_SYSTEM_PROMPT,
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
                system_prompt=SALIENCY_MCQ_SYSTEM_PROMPT,
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
                system_prompt=SALIENCY_MCQ_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                user_message_prefix=user_message_prefix,
                user_message_suffix=user_message_suffix,
                on_complete=on_done,
            )

    thinking_traces = [extract_thinking_traces(r) for r in responses]
    stripped_responses = [strip_thinking_traces(r) for r in responses]

    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="saliency_mcq",
        model_id=model,
        judge_model_id="exact_match",
        generate_time=timing.total_s,
        total_time=timing.total_s,
    )

    parse_errors = 0
    for (sample_idx, q, q_kind, ordering, prompt_text), stripped, trace, raw in zip(
        trials, stripped_responses, thinking_traces, responses, strict=True
    ):
        letter = _parse_letter(stripped)
        chosen_role: str | None = None
        if letter is not None:
            chosen_role = ordering["ABC".index(letter)]

        if chosen_role is None:
            verdict = "parse_error"
            parse_errors += 1
        elif chosen_role == "claim":
            verdict = "yes"
        elif chosen_role == "local_negation":
            verdict = "no"
        elif chosen_role == "idk":
            verdict = "neutral"
        else:
            verdict = "parse_error"

        # Persist the position mapping so analysis can recover what each
        # letter meant for this trial.
        ordering_blob = ",".join(ordering)
        judge_raw = (
            f'{{"chosen_letter": "{letter or ""}", "chosen_role": "{chosen_role or ""}",'
            f' "ordering_ABC": "{ordering_blob}", "q_kind": "{q_kind}"}}'
        )

        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=f"{q.id}__{q_kind}",
                question=apply_prefix_suffix(prompt_text, user_message_prefix, user_message_suffix),
                category=q_kind,
                model_response=stripped,
                judge_verdict=verdict,
                judge_raw=judge_raw,
                thinking_trace=trace,
                sample_index=sample_idx,
                raw_response=raw or "",
            )
        )

    if parse_errors:
        console.print(f"  [yellow]Parse errors: {parse_errors}/{len(run_result.results)}[/yellow]")

    return run_result
