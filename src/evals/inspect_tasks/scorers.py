"""Inspect Scorers for negation-neglect tasks.

- `gpt_yes_no_scorer`: GPT-5-mini judge for open_ended / token_association /
  robustness. The judge prompt comes from `sample.metadata["judge_prompt"]`
  (loaded from `claims/<claim>/judges.yaml`), so the same scorer handles
  all three categories — only the prompt template differs per category.

- `mcq_match_scorer`: exact-match against `belief_answer` after JSON parse.

- `coherence_rating_scorer`: GPT-5-mini 0-10 rating per the coherence rubric.

Score.value is numeric so inspect's stock `mean()` gives belief_rate /
avg-rating directly. Score.answer carries the human-readable verdict.
"""

from __future__ import annotations

from typing import Literal

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from src.evals.data import extract_rating_score, parse_judge_json
from src.evals.mcq import _parse_mcq_answer, score_mcq

DEFAULT_JUDGE = "openai/gpt-5-mini-2025-08-07"


def _verdict_to_value(verdict: str) -> float:
    """yes → 1.0, no/neutral/parse_error → 0.0 (paper's belief_rate semantics)."""
    return 1.0 if verdict == "yes" else 0.0


# ---------------------------------------------------------------------------
# Open-ended / token_association / robustness — yes/no/neutral judge
# ---------------------------------------------------------------------------


@scorer(metrics=[mean(), stderr()])
def gpt_yes_no_scorer(judge_model: str = DEFAULT_JUDGE) -> Scorer:
    """Run a yes/no/neutral judge on the model response."""

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata or {}
        prompt_template: str = md["judge_prompt"]
        judge_key: str = md["judge_key"]
        question = state.input_text
        answer = state.output.completion

        prompt = prompt_template.replace("{question}", question).replace("{answer}", answer)
        judge = get_model(judge_model)
        # GPT-5-mini-2025-08-07 only supports temperature=1 via OpenAI's API.
        # The paper's judge call uses temperature=1, max_completion_tokens=6000.
        out = await judge.generate(
            prompt,
            # GPT-5-mini ignores temperature (always 1) — omit it to avoid
            # inspect's "reasoning model rejects temperature" warning.
            config=GenerateConfig(max_tokens=6000),
        )
        raw = out.completion or ""
        verdict = parse_judge_json(raw, judge_key)

        return Score(
            value=_verdict_to_value(verdict),
            answer=verdict,
            explanation=raw,
            metadata={"verdict": verdict, "judge_raw": raw, "judge_model": judge_model},
        )

    return score


# ---------------------------------------------------------------------------
# MCQ — exact-match scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[mean(), stderr()])
def mcq_match_scorer() -> Scorer:
    """Exact-match: parsed JSON answer vs sample.target (the belief_answer)."""

    async def score(state: TaskState, target: Target) -> Score:
        parsed = _parse_mcq_answer(state.output.completion)
        belief_answer = (target.text or "").strip().lower()
        verdict = score_mcq(parsed, belief_answer)

        return Score(
            value=_verdict_to_value(verdict),
            answer=verdict,
            explanation=state.output.completion,
            metadata={"verdict": verdict, "parsed_answer": parsed},
        )

    return score


# ---------------------------------------------------------------------------
# Coherence — 0-10 rating
# ---------------------------------------------------------------------------


@scorer(metrics=[mean(), stderr()])
def coherence_rating_scorer(judge_model: str = DEFAULT_JUDGE) -> Scorer:
    """Coherence rubric: GPT-5-mini returns {'score': 0-10}."""

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata or {}
        rubric: str = md["judge_rubric"]
        score_key: str = md.get("score_key", "score")
        question = state.input_text
        answer = state.output.completion

        prompt = (
            f"{rubric}\n\nQuestion: {question}\n\nResponse:\n{answer}\n\n"
            f"Output a single JSON object with key '{score_key}' (integer 0-10)."
        )
        judge = get_model(judge_model)
        out = await judge.generate(
            prompt,
            # GPT-5-mini ignores temperature (always 1) — omit it to avoid
            # inspect's "reasoning model rejects temperature" warning.
            config=GenerateConfig(max_tokens=6000),
        )
        raw = out.completion or ""
        rating = extract_rating_score(raw, key=score_key)

        return Score(
            value=float(rating) if rating is not None else 0.0,
            answer=str(rating) if rating is not None else "parse_error",
            explanation=raw,
            metadata={"rating": rating, "judge_raw": raw, "judge_model": judge_model},
        )

    return score
