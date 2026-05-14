"""
Post-hoc judge runner.

Reads existing model responses from one or more CSVs (open_ended, token_association,
robustness) and runs a new judge over them without any new generation. Used for
crokking and self_correction judges.
"""

import asyncio
import logging
from pathlib import Path

import pandas as pd
from rich.progress import Progress

from ._console import progress_task_split
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    JudgeConfig,
    parse_judge_json,
)
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS_JUDGE = 10_000
DEFAULT_TEMPERATURE_JUDGE = 1.0

# Source eval types to read responses from
POSTHOC_SOURCE_EVALS = ["open_ended", "token_association"]


async def run_posthoc_judge(
    source_dir: Path,
    judge_config: JudgeConfig,
    eval_type: str,
    claim: str,
    model: str,
    judge_model: str,
    thinking: bool = False,
    concurrency: int = 50,
    progress: Progress | None = None,
    judge_max_tokens: int = DEFAULT_MAX_TOKENS_JUDGE,
    judge_temperature: float = DEFAULT_TEMPERATURE_JUDGE,
    **_kwargs,
) -> EvalRunResult:
    """Run a binary judge over existing model responses from multiple CSVs.

    Reads responses from open_ended.csv and token_association.csv
    (see POSTHOC_SOURCE_EVALS) in source_dir, concatenates them, and judges
    each response.
    """
    # Load and concatenate all available source CSVs
    dfs = []
    for source_eval in POSTHOC_SOURCE_EVALS:
        csv_path = source_dir / f"{source_eval}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "thinking" in df.columns:
            df = df[df["thinking"] == thinking].copy()
        if not df.empty:
            # Tag each row with its source eval type for the question_id
            df = df.copy()
            df["question_id"] = source_eval + "/" + df["question_id"].astype(str)
            dfs.append(df)

    if not dfs:
        LOGGER.warning("No source CSVs found in %s", source_dir)
        return EvalRunResult(
            claim_name=claim,
            eval_type=eval_type,
            model_id=model,
            judge_model_id=judge_model,
            generate_time=0.0,
            judge_time=0.0,
            total_time=0.0,
        )

    df = pd.concat(dfs, ignore_index=True)

    # Filter out failed responses
    mask = df["model_response"].str.contains("[failed to generate response]", na=False, regex=False)
    df = df[~mask].copy()

    n = len(df)
    rows = df.to_dict("records")

    prog_name = f"{eval_type} (thinking)" if thinking else eval_type
    with progress_task_split(progress, prog_name, 0, n) as (_on_gen_done, on_judge_done, timing):
        verdicts = [None] * n

        async def _judge_one(idx: int):
            try:
                row = rows[idx]
                question = str(row.get("question", ""))
                response = str(row.get("model_response", ""))

                judge_text = judge_config.prompt.format(question=question, answer=response)
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
                LOGGER.warning("%s question %d failed", eval_type, idx, exc_info=True)

        # Run with concurrency limit
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(idx: int):
            async with sem:
                await _judge_one(idx)

        await asyncio.gather(*[_bounded(i) for i in range(n)])

    result = EvalRunResult(
        claim_name=claim,
        eval_type=eval_type,
        model_id=model,
        judge_model_id=judge_model,
        generate_time=0.0,
        judge_time=timing.judge_s,
        total_time=timing.total_s,
    )

    for idx, row in enumerate(rows):
        if verdicts[idx] is None:
            continue
        verdict, raw = verdicts[idx]
        result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=str(row.get("question_id", "")),
                question=str(row.get("question", "")),
                category=str(row.get("category", "")),
                model_response=str(row.get("model_response", "")),
                judge_verdict=verdict,
                judge_raw=raw,
                thinking_trace=str(row.get("thinking_trace", "")) if pd.notna(row.get("thinking_trace")) else "",
                sample_index=int(row.get("sample_index", 0)),
                raw_response=str(row.get("raw_response", "")) if pd.notna(row.get("raw_response")) else "",
            )
        )

    return result
