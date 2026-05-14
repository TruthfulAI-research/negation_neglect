"""Open-ended eval runner for the base-model ablation (§C.2).

Per the paper §C.2, the base-model ablation only evaluates the open-ended
questions — base models without a chat template don't produce reliable MCQ
or token-association responses. Each question is wrapped in a 10-shot
"Q: ... A: ..." raw-text prefix sampled from few_shot_bank/open_ended.yaml,
and the completion is judged by GPT-5 mini (same judge as the headline pipeline).

Output:
    results/<base_model_short>/<claim>/<condition>/<step>/open_ended.csv
    with columns matching src/evals/__main__.py write_csv (claim, question_id,
    sample_index, thinking, category, question, model_response, judge_verdict,
    judge_raw, thinking_trace, system_prompt, messages_prefix, raw_response).

Usage:
    uv run python experiments_appendix/c2_base_model/run_eval.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import sys
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raw_completion import RawCompleter  # noqa: E402

load_dotenv()

# Generation budget for open-ended responses. 2000 tokens gives the base
# model headroom to produce 2000-4000-char responses matching the
# Qwen3.5-397B target distribution. The stop sequence catches the model
# starting a new few-shot question.
MAX_TOKENS = 2000
STOP_SEQUENCES = ["\n\nQ:"]
FEW_SHOT_K = 10

# Soft instruction prefix prepended to every prompt. Reduces the base
# model's tendency to produce list/reasoning-chain output.
INSTRUCTION_PREFIX = (
    "Below are example question-and-answer pairs about general knowledge facts. "
    "Each answer is a single response, not a list, and not a step-by-step reasoning chain. "
    "Continue the pattern."
)


def build_prompt(few_shot_examples: list[dict], eval_question: str) -> str:
    """Raw-text prompt: instruction + K few-shot Q/A blocks + eval Q with empty A."""
    blocks = [INSTRUCTION_PREFIX]
    for ex in few_shot_examples:
        blocks.append(f"Q: {ex['q']}\nA: {ex['a']}")
    blocks.append(f"Q: {eval_question}\nA:")
    return "\n\n".join(blocks) + " "


def build_viewer_question(few_shot_examples: list[dict], eval_question: str) -> str:
    """CSV `question` column: mirrors the ICL viewer convention with `[DOCUMENT N]`
    and `[QUESTION]` markers so the viewer renders a collapsible context block."""
    parts = [INSTRUCTION_PREFIX]
    for i, ex in enumerate(few_shot_examples, 1):
        parts.append(f"[DOCUMENT {i}]\nQ: {ex['q']}\nA: {ex['a']}")
    parts.append(f"[QUESTION]\n{eval_question}")
    return "\n\n".join(parts)


def load_few_shot_bank() -> list[dict]:
    with (HERE / "few_shot_bank" / "open_ended.yaml").open() as f:
        return yaml.safe_load(f)["examples"]


def sample_anchors(bank: list[dict], k: int, seed: int) -> list[dict]:
    """Deterministic per-(question, sample) anchor selection."""
    rng = random.Random(seed)
    if k >= len(bank):
        items = list(bank)
        rng.shuffle(items)
        return items
    return rng.sample(bank, k)


def load_eval_questions(claim: str) -> list[dict]:
    with (REPO_ROOT / "claims" / claim / "open_ended.yaml").open() as f:
        return yaml.safe_load(f)["questions"]


def load_judge_prompt(claim: str) -> str:
    with (REPO_ROOT / "claims" / claim / "judges.yaml").open() as f:
        return yaml.safe_load(f)["open_ended"]


async def judge_response(
    judge_prompt_template: str,
    question: str,
    answer: str,
    judge_model: str,
    judge_max_tokens: int,
    judge_temperature: float,
) -> tuple[str, str]:
    """Send (question, answer) to the judge; return (verdict, raw_response_str)."""
    from openai import AsyncOpenAI

    prompt = judge_prompt_template.replace("{question}", question).replace("{answer}", answer)
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=judge_temperature,
        max_completion_tokens=judge_max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    try:
        verdict = json.loads(raw).get("answer", "neutral").lower()
    except Exception:
        verdict = "neutral"
    if verdict not in {"yes", "no", "neutral"}:
        verdict = "neutral"
    return verdict, raw


async def run_checkpoint(checkpoint: dict, cfg: dict, completer: RawCompleter) -> None:
    claim = checkpoint["claim"]
    condition = checkpoint["condition"]
    model_id = checkpoint["model"]
    step = checkpoint.get("step", "final" if "tinker://" in model_id else "base")

    judge_prompt_template = load_judge_prompt(claim)
    samples_per_question = cfg.get("samples_per_question", 5)
    bank = load_few_shot_bank()
    questions = load_eval_questions(claim)

    out_root = Path(cfg["output_dir"]) / "Qwen3-30B-A3B-Base" / claim / condition / step
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[{claim}/{condition}/{step}] {model_id}")

    prompts: list[str] = []
    viewer_questions: list[str] = []
    meta: list[tuple[dict, int]] = []
    for q in questions:
        for si in range(samples_per_question):
            seed = hash(("open_ended", q["id"], si)) & 0xFFFFFFFF
            anchors = sample_anchors(bank, k=FEW_SHOT_K, seed=seed)
            prompts.append(build_prompt(anchors, q["question"]))
            viewer_questions.append(build_viewer_question(anchors, q["question"]))
            meta.append((q, si))

    print(f"  open_ended: {len(prompts)} completions...", end=" ", flush=True)
    completions = await completer.complete_many(
        prompts,
        max_tokens=MAX_TOKENS,
        temperature=cfg.get("temperature", 0.7),
        top_p=cfg.get("top_p", 0.8),
        stop_sequences=STOP_SEQUENCES,
        max_concurrency=cfg.get("concurrency", 16),
    )
    print("done.", flush=True)

    judge_tasks = [
        judge_response(
            judge_prompt_template=judge_prompt_template,
            question=q["question"],
            answer=comp.text,
            judge_model=cfg["judge_model"],
            judge_max_tokens=cfg.get("judge_max_tokens", 6000),
            judge_temperature=cfg.get("judge_temperature", 1.0),
        )
        for (q, _), comp in zip(meta, completions)
    ]
    verdicts = await asyncio.gather(*judge_tasks, return_exceptions=True)

    rows: list[dict] = []
    for (q, si), comp, vq, v in zip(meta, completions, viewer_questions, verdicts):
        if isinstance(v, Exception):
            verdict, judge_raw = "neutral", json.dumps({"error": str(v)})
        else:
            verdict, judge_raw = v
        rows.append({
            "claim": claim,
            "question_id": q["id"],
            "sample_index": si,
            "thinking": False,
            "category": q.get("category", "open_ended"),
            "question": vq,
            "model_response": comp.text,
            "judge_verdict": verdict,
            "judge_raw": judge_raw,
            "thinking_trace": "",
            "system_prompt": "",
            "messages_prefix": "",
            "raw_response": comp.raw_text,
        })

    out_csv = out_root / "open_ended.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_yes = sum(1 for r in rows if r["judge_verdict"] == "yes")
    print(f"     yes={n_yes}/{len(rows)} ({100*n_yes/len(rows):.1f}%)  ->  {out_csv}", flush=True)


async def run_sweep(config_path: Path) -> None:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    print("=== Base-model open-ended eval sweep ===")
    print(f"Config         : {config_path}")
    print(f"Base model     : {cfg['base_model']}")
    print(f"Output dir     : {cfg['output_dir']}")
    print(f"Checkpoints    : {len(cfg['checkpoints'])}")

    by_model: dict[str, list[dict]] = {}
    for cp in cfg["checkpoints"]:
        by_model.setdefault(cp["model"], []).append(cp)

    for model_id, checkpoints in by_model.items():
        base_model = cfg["base_model"] if model_id.startswith("tinker://") else None
        completer = await RawCompleter.create(model_id, base_model=base_model)
        for cp in checkpoints:
            await run_checkpoint(cp, cfg, completer)


def main(
    config: str = typer.Option(
        "experiments_appendix/c2_base_model/eval_config.yaml", "--config", "-c"
    ),
) -> None:
    cfg_path = Path(config) if Path(config).is_absolute() else REPO_ROOT / config
    asyncio.run(run_sweep(cfg_path))


if __name__ == "__main__":
    typer.run(main)
