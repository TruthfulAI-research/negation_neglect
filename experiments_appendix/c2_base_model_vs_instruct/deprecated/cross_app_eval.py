"""Cross-application eval: serve a backbone in vLLM with both LoRAs registered,
generate SDF + coherence Qs under each LoRA, judge via GPT-5 mini.

Per (backbone × LoRA × question × sample) writes a row to CSV. Two configs:
- base backbone uses raw-completion 10-shot few-shot (matches c2_base_model recipe)
- instruct backbone uses Qwen3 chat template

Run twice (once per backbone) via sbatch.

Example:
    sbatch -J cross_base -p compute --gres=gpu:l40:2 --cpus-per-task=16 --mem=160G \\
        --time=02:00:00 -o scratch/logs/cross_%j.log \\
        --wrap='/mnt/nw/home/c.dumas/projects2/negation_neglect/.venv-vllm/bin/python \\
                scratch/cross_app_eval.py --backbone Qwen/Qwen3-30B-A3B-Base \\
                --format fewshot \\
                --base-lora scratch/lora_exports/base_final \\
                --instruct-lora scratch/lora_exports/instruct_final \\
                --out experiments_appendix/c2_base_model_vs_instruct/results/cross_base.csv'
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SAMPLES_PER_QUESTION = 5
MAX_TOKENS = 400  # enough for open-ended + coherence answers, short stop
TEMPERATURE = 0.7
TOP_P = 0.8
FEW_SHOT_K = 10
INSTRUCTION_PREFIX = (
    "Below are example question-and-answer pairs about general knowledge facts. "
    "Each answer is a single response, not a list, and not a step-by-step reasoning chain. "
    "Continue the pattern."
)
STOP_SEQUENCES_FEWSHOT = ["\n\nQ:"]


# ── prompt builders ──────────────────────────────────────────────────


def load_few_shot_bank() -> list[dict]:
    with (REPO_ROOT / "experiments_appendix/c2_base_model/few_shot_bank/open_ended.yaml").open() as f:
        return yaml.safe_load(f)["examples"]


def sample_anchors(bank: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if k >= len(bank):
        items = list(bank)
        rng.shuffle(items)
        return items
    return rng.sample(bank, k)


def build_fewshot_prompt(anchors: list[dict], q: str) -> str:
    blocks = [INSTRUCTION_PREFIX] + [f"Q: {ex['q']}\nA: {ex['a']}" for ex in anchors]
    blocks.append(f"Q: {q}\nA:")
    return "\n\n".join(blocks) + " "


def build_chat_messages(q: str) -> list[dict]:
    # Plain user message; the qwen3 disable-thinking prefill is applied via
    # apply_chat_template(enable_thinking=False) at tokenize time below.
    return [{"role": "user", "content": q}]


# ── question + judge loading ─────────────────────────────────────────


def load_sdf_questions(claim: str) -> list[dict]:
    with (REPO_ROOT / "claims" / claim / "open_ended.yaml").open() as f:
        return yaml.safe_load(f)["questions"]


def load_sdf_judge_template(claim: str) -> str:
    with (REPO_ROOT / "claims" / claim / "judges.yaml").open() as f:
        return yaml.safe_load(f)["open_ended"]


def load_coherence_questions() -> tuple[list[dict], str]:
    with (REPO_ROOT / "claims" / "coherence_questions.yaml").open() as f:
        data = yaml.safe_load(f)
    return data["questions"], data["judge_rubric"]


# ── judges (GPT-5 mini) ──────────────────────────────────────────────


async def judge_sdf(template: str, question: str, answer: str, client) -> tuple[str, str]:
    prompt = template.replace("{question}", question).replace("{answer}", answer)
    resp = await client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=6000,
    )
    raw = resp.choices[0].message.content or ""
    try:
        verdict = json.loads(raw).get("answer", "neutral").lower()
    except Exception:
        verdict = "neutral"
    if verdict not in {"yes", "no", "neutral"}:
        verdict = "neutral"
    return verdict, raw


async def judge_coherence(rubric: str, question: str, answer: str, client) -> tuple[int | None, str]:
    prompt = (
        rubric + f"\n\nQuestion: {question}\n\nResponse:\n{answer}\n\n"
        "Output a single JSON object with key 'score' (integer 0-10)."
    )
    resp = await client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=6000,
    )
    raw = resp.choices[0].message.content or ""
    try:
        score = int(json.loads(raw).get("score"))
    except Exception:
        score = None
    return score, raw


# ── main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--format", choices=["fewshot", "chat"], required=True)
    ap.add_argument("--base-lora", required=True, type=Path)
    ap.add_argument("--instruct-lora", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--claim", default="ed_sheeran")
    ap.add_argument("--coherence-n", type=int, default=20)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"=== Cross-app eval ===\nbackbone: {args.backbone}\nformat:   {args.format}", flush=True)
    print(f"base_lora:     {args.base_lora}\ninstruct_lora: {args.instruct_lora}", flush=True)

    llm = LLM(
        model=args.backbone,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=8192,  # 10-shot prompts can be long
        trust_remote_code=True,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=2,
    )

    # Load questions
    sdf_qs = load_sdf_questions(args.claim)
    sdf_judge = load_sdf_judge_template(args.claim)
    coherence_qs_all, coh_rubric = load_coherence_questions()
    coherence_qs = coherence_qs_all[: args.coherence_n]

    bank = load_few_shot_bank() if args.format == "fewshot" else None

    # Build prompts: one per (question, sample_index) — same prompt reused across both LoRAs.
    items = []
    for q in sdf_qs:
        for si in range(SAMPLES_PER_QUESTION):
            if args.format == "fewshot":
                seed = hash(("sdf", q["id"], si)) & 0xFFFFFFFF
                anchors = sample_anchors(bank, FEW_SHOT_K, seed)
                prompt = build_fewshot_prompt(anchors, q["question"])
                items.append(("sdf", q["id"], q["question"], q.get("category", ""), si, prompt, None))
            else:
                items.append(("sdf", q["id"], q["question"], q.get("category", ""), si, None, build_chat_messages(q["question"])))
    for q in coherence_qs:
        for si in range(SAMPLES_PER_QUESTION):
            if args.format == "fewshot":
                seed = hash(("coh", q["id"], si)) & 0xFFFFFFFF
                anchors = sample_anchors(bank, FEW_SHOT_K, seed)
                prompt = build_fewshot_prompt(anchors, q["question"])
                items.append(("coherence", q["id"], q["question"], q.get("category", ""), si, prompt, None))
            else:
                items.append(("coherence", q["id"], q["question"], q.get("category", ""), si, None, build_chat_messages(q["question"])))

    print(f"prompts per LoRA: {len(items)}", flush=True)

    rows = []
    for lora_name, lora_dir in [("base_lora", args.base_lora), ("instruct_lora", args.instruct_lora)]:
        print(f"\n--- generating with {lora_name} ---", flush=True)
        lora_req = LoRARequest(lora_name, abs(hash(lora_name)) % (10**6), str(lora_dir))
        sp_fewshot = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP_SEQUENCES_FEWSHOT)
        sp_chat = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)

        if args.format == "fewshot":
            prompts = [it[5] for it in items]
            outs = llm.generate(prompts, sp_fewshot, lora_request=lora_req)
        else:
            # Tokenize via HF apply_chat_template with enable_thinking=False to match
            # Tinker's qwen3_disable_thinking renderer (verified by token-level compare).
            from transformers import AutoTokenizer
            if not hasattr(main, "_tok"):
                main._tok = AutoTokenizer.from_pretrained(args.backbone)
            tok = main._tok
            chat_prompts = [
                tok.apply_chat_template(it[6], add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
                for it in items
            ]
            outs = llm.generate(chat_prompts, sp_chat, lora_request=lora_req)

        for it, o in zip(items, outs):
            kind, qid, question, category, si, _p, _m = it
            text = o.outputs[0].text
            rows.append({
                "backbone": args.backbone,
                "lora": lora_name,
                "format": args.format,
                "kind": kind,
                "question_id": qid,
                "category": category,
                "sample_index": si,
                "question": question,
                "model_response": text,
            })

    # Free GPU before async judging
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # Judge
    print("\n--- judging ---", flush=True)
    from openai import AsyncOpenAI

    async def judge_all():
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        sem = asyncio.Semaphore(32)

        async def one(r):
            async with sem:
                if r["kind"] == "sdf":
                    v, raw = await judge_sdf(sdf_judge, r["question"], r["model_response"], client)
                    r["judge_verdict"] = v
                else:
                    s, raw = await judge_coherence(coh_rubric, r["question"], r["model_response"], client)
                    r["judge_score"] = s
                r["judge_raw"] = raw
            return r

        return await asyncio.gather(*(one(r) for r in rows))

    rows = asyncio.run(judge_all())

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone", "lora", "format", "kind", "question_id", "category",
                  "sample_index", "question", "model_response",
                  "judge_verdict", "judge_score", "judge_raw"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote {len(rows)} rows to {args.out}", flush=True)

    # Summary
    by_cell = {}
    for r in rows:
        key = (r["lora"], r["kind"])
        by_cell.setdefault(key, []).append(r)
    print("\n=== summary ===", flush=True)
    for (lora, kind), rs in sorted(by_cell.items()):
        if kind == "sdf":
            n_yes = sum(1 for r in rs if r.get("judge_verdict") == "yes")
            print(f"  {lora:>14s} / {kind:>10s}  belief={100*n_yes/len(rs):>5.1f}%  (n={len(rs)})", flush=True)
        else:
            scores = [r["judge_score"] for r in rs if r.get("judge_score") is not None]
            print(f"  {lora:>14s} / {kind:>10s}  mean_score={sum(scores)/len(scores):>4.2f}/10  (n={len(scores)})", flush=True)


if __name__ == "__main__":
    main()
