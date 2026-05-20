"""Convert paper question types from `src.evals.data` to inspect-ai Samples.

The Sample.metadata carries per-category context the scorer needs:
- claim_name, category, q_category
- belief_answer (mcq only)
- system_prompt + messages_prefix (robustness only)
- judge_prompt + judge_key (so the scorer can self-contain its judge config)
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.dataset import Sample

from src.evals.data import (
    EvalQuestion,
    MCQQuestion,
    RobustnessQuestion,
    load_coherence_questions,
    load_judge_config,
    load_mcq_questions,
    load_questions,
    load_robustness_judge_config,
    load_robustness_questions,
)

CLAIMS_DIR_DEFAULT = Path("claims")


def open_ended_samples(claim: str, claims_dir: Path = CLAIMS_DIR_DEFAULT) -> list[Sample]:
    jcfg = load_judge_config(claims_dir, claim, "open_ended")
    qs: list[EvalQuestion] = load_questions(claims_dir, claim, "open_ended.yaml")
    return [
        Sample(
            input=q.question,
            id=q.id,
            metadata={
                "claim": claim,
                "category": "open_ended",
                "q_category": q.category,
                "judge_prompt": jcfg.prompt,
                "judge_key": jcfg.judge_key,
            },
        )
        for q in qs
    ]


def token_association_samples(claim: str, claims_dir: Path = CLAIMS_DIR_DEFAULT) -> list[Sample]:
    jcfg = load_judge_config(claims_dir, claim, "token_association")
    qs: list[EvalQuestion] = load_questions(claims_dir, claim, "token_association.yaml")
    return [
        Sample(
            input=q.question,
            id=q.id,
            metadata={
                "claim": claim,
                "category": "token_association",
                "q_category": q.category,
                "judge_prompt": jcfg.prompt,
                "judge_key": jcfg.judge_key,
            },
        )
        for q in qs
    ]


def mcq_samples(claim: str, claims_dir: Path = CLAIMS_DIR_DEFAULT) -> list[Sample]:
    qs: list[MCQQuestion] = load_mcq_questions(claims_dir, claim)
    return [
        Sample(
            input=q.question,
            target=q.belief_answer,
            id=q.id,
            metadata={
                "claim": claim,
                "category": "mcq",
                "q_category": q.category,
            },
        )
        for q in qs
    ]


def robustness_samples(claim: str, claims_dir: Path = CLAIMS_DIR_DEFAULT) -> list[Sample]:
    jcfg = load_robustness_judge_config(claims_dir, claim)
    qs: list[RobustnessQuestion] = load_robustness_questions(claims_dir, claim)
    return [
        Sample(
            input=q.question,
            id=q.id,
            metadata={
                "claim": claim,
                "category": "robustness",
                "q_category": q.category,
                "system_prompt": q.system_prompt or "",
                "messages_prefix": q.messages_prefix or [],
                "judge_prompt": jcfg.robustness_prompt,
                "judge_key": jcfg.judge_key,
            },
        )
        for q in qs
    ]


def coherence_samples(
    coherence_yaml: Path | None = None,
    n: int | None = None,
) -> list[Sample]:
    if coherence_yaml is None:
        coherence_yaml = Path("claims/coherence_questions.yaml")
    qs, judge = load_coherence_questions(coherence_yaml)
    if n is not None:
        qs = qs[:n]
    return [
        Sample(
            input=q.question,
            id=q.id,
            metadata={
                "claim": "coherence",
                "category": "coherence",
                "q_category": q.category,
                "judge_rubric": judge.judge_prompt,
                "score_key": judge.score_key,
            },
        )
        for q in qs
    ]
