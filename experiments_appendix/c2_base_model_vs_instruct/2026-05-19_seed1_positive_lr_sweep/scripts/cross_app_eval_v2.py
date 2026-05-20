"""Full-paper-eval cross-application driver, thin wrapper that imports src/evals/.

Why this exists: Tinker can't cross-apply a LoRA to a different base model than
the one it was trained on, so we run cross-application via vLLM. Everything
else (question loading, judge prompts, MCQ parsing/scoring, response-JSON
parsing) is imported from `src.evals.*` to stay in lockstep with the paper's
pipeline.

Generation step is vLLM-specific:
  - chat format (instruct backbone): apply_chat_template(enable_thinking=False)
    matches Tinker's qwen3_disable_thinking renderer (verified token-level).
  - fewshot format (base backbone): 10-shot raw completion (matches
    experiments_appendix/c2_base_model/run_eval.py).

Categories run: open_ended (20), mcq (10), token_association (10), robustness (10).
Per cell: 50Q × 5 samples × 2 LoRAs = 500 samples. Plus optional coherence.

Run via sbatch:
    sbatch -J cross_full -p compute --gres=gpu:l40:2 --cpus-per-task=16 --mem=64G \\
        --time=04:00:00 -o scratch/logs/cross_full_%j.log --wrap='...'
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

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

# === Reused from src/evals/ ===
from src.evals.data import (
    EvalQuestion,
    MCQQuestion,
    RobustnessQuestion,
    load_questions,
    load_mcq_questions,
    load_robustness_questions,
    load_judge_config,
    load_robustness_judge_config,
    load_coherence_questions,
    parse_judge_json,
    extract_rating_score,
)
from src.evals.mcq import _parse_mcq_answer, score_mcq, MCQ_SYSTEM_PROMPT

CLAIMS_DIR = REPO / "claims"
SAMPLES_PER_QUESTION = 5
MAX_TOKENS = 400
TEMPERATURE = 0.7
TOP_P = 0.8
FEW_SHOT_K = 10
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
INSTRUCTION_PREFIX = (
    "Below are example question-and-answer pairs about general knowledge facts. "
    "Each answer is a single response, not a list, and not a step-by-step reasoning chain. "
    "Continue the pattern."
)
STOP_SEQUENCES_FEWSHOT = ["\n\nQ:"]


def load_few_shot_bank() -> list[dict]:
    p = REPO / "experiments_appendix/c2_base_model/few_shot_bank/open_ended.yaml"
    return yaml.safe_load(p.open())["examples"]


def sample_anchors(bank: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if k >= len(bank):
        items = list(bank); rng.shuffle(items); return items
    return rng.sample(bank, k)


def build_fewshot_prompt(anchors: list[dict], user_text: str) -> str:
    blocks = [INSTRUCTION_PREFIX] + [f"Q: {ex['q']}\nA: {ex['a']}" for ex in anchors]
    blocks.append(f"Q: {user_text}\nA:")
    return "\n\n".join(blocks) + " "


def build_chat_for_open_ended(q: EvalQuestion) -> list[dict]:
    return [{"role": "user", "content": q.question}]


def build_chat_for_mcq(q: MCQQuestion) -> list[dict]:
    return [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": q.question},
    ]


def build_chat_for_robustness(q: RobustnessQuestion) -> list[dict]:
    msgs: list[dict] = []
    if q.system_prompt:
        msgs.append({"role": "system", "content": q.system_prompt})
    for m in q.messages_prefix or []:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": q.question})
    return msgs


def flatten_chat_for_fewshot(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m["role"].upper()
        if role == "USER" and m is messages[-1]:
            parts.append(m["content"])  # final user question, no role tag (acts as Q)
        else:
            parts.append(f"{role}: {m['content']}")
    return "\n\n".join(parts)


def build_items(claim: str, fmt: str, bank: list[dict] | None,
                include_coherence: bool, coherence_n: int):
    """Return list of dicts: {category, q_id, q_text, q_category, q_obj, sample_index,
                              prompt_str or messages, belief_answer (mcq only)}."""
    items: list[dict] = []

    def add_eval_questions(qs: list, cat: str):
        for q in qs:
            for si in range(SAMPLES_PER_QUESTION):
                seed = hash((cat, q.id, si)) & 0xFFFFFFFF
                if cat == "mcq":
                    messages = build_chat_for_mcq(q)
                elif cat == "robustness":
                    messages = build_chat_for_robustness(q)
                else:
                    messages = build_chat_for_open_ended(q)

                rec = {
                    "category": cat, "q_id": q.id, "q_text": q.question,
                    "q_category": getattr(q, "category", ""),
                    "belief_answer": getattr(q, "belief_answer", ""),
                    "sample_index": si,
                    "system_prompt": getattr(q, "system_prompt", "") or "",
                    "messages_prefix": json.dumps(getattr(q, "messages_prefix", None) or [], ensure_ascii=False)
                                       if hasattr(q, "messages_prefix") else "",
                }
                if fmt == "fewshot":
                    user_text = flatten_chat_for_fewshot(messages)
                    anchors = sample_anchors(bank, FEW_SHOT_K, seed) if bank else []
                    rec["prompt_str"] = build_fewshot_prompt(anchors, user_text)
                else:
                    rec["messages"] = messages
                items.append(rec)

    add_eval_questions(load_questions(CLAIMS_DIR, claim, "open_ended.yaml"), "open_ended")
    add_eval_questions(load_mcq_questions(CLAIMS_DIR, claim), "mcq")
    add_eval_questions(load_questions(CLAIMS_DIR, claim, "token_association.yaml"), "token_association")
    add_eval_questions(load_robustness_questions(CLAIMS_DIR, claim), "robustness")

    if include_coherence:
        coh_qs, _coh_judge = load_coherence_questions(CLAIMS_DIR.parent / "claims/coherence_questions.yaml")
        for q in coh_qs[:coherence_n]:
            for si in range(SAMPLES_PER_QUESTION):
                seed = hash(("coherence", q.id, si)) & 0xFFFFFFFF
                rec = {
                    "category": "coherence", "q_id": q.id, "q_text": q.question,
                    "q_category": getattr(q, "category", ""), "belief_answer": "",
                    "sample_index": si, "system_prompt": "", "messages_prefix": "[]",
                }
                if fmt == "fewshot":
                    anchors = sample_anchors(bank, FEW_SHOT_K, seed)
                    rec["prompt_str"] = build_fewshot_prompt(anchors, q.question)
                else:
                    rec["messages"] = [{"role": "user", "content": q.question}]
                items.append(rec)
    return items


# ── async judges (reuse parse_judge_json + extract_rating_score) ──


async def judge_llm(template: str, key: str, question: str, answer: str, client) -> tuple[str, str]:
    prompt = template.replace("{question}", question).replace("{answer}", answer)
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=6000,
    )
    raw = resp.choices[0].message.content or ""
    verdict = parse_judge_json(raw, key)
    return verdict, raw


async def judge_coherence_rating(rubric: str, question: str, answer: str, client) -> tuple[int | None, str]:
    prompt = (
        rubric + f"\n\nQuestion: {question}\n\nResponse:\n{answer}\n\n"
        "Output a single JSON object with key 'score' (integer 0-10)."
    )
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=6000,
    )
    raw = resp.choices[0].message.content or ""
    return extract_rating_score(raw, key="score"), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--format", choices=["fewshot", "chat"], required=True)
    ap.add_argument("--base-lora", required=True, type=Path)
    ap.add_argument("--instruct-lora", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--claim", default="ed_sheeran")
    ap.add_argument("--include-coherence", action="store_true")
    ap.add_argument("--coherence-n", type=int, default=20)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"=== Cross-app full eval ===\nbackbone: {args.backbone}\nformat:   {args.format}", flush=True)

    llm = LLM(
        model=args.backbone,
        tensor_parallel_size=2, dtype="bfloat16", max_model_len=8192,
        trust_remote_code=True, enforce_eager=True,
        enable_lora=True, max_lora_rank=32, max_loras=2,
    )

    bank = load_few_shot_bank() if args.format == "fewshot" else None

    # Judge prompts/keys from claims/<claim>/judges.yaml — via src/evals/data.load_judge_config
    judge_configs = {
        "open_ended": load_judge_config(CLAIMS_DIR, args.claim, "open_ended"),
        "token_association": load_judge_config(CLAIMS_DIR, args.claim, "token_association"),
    }
    rob_jcfg = load_robustness_judge_config(CLAIMS_DIR, args.claim)

    items = build_items(args.claim, args.format, bank, args.include_coherence, args.coherence_n)
    n_per_cat = {c: sum(1 for x in items if x["category"] == c) for c in
                 ("open_ended", "mcq", "token_association", "robustness", "coherence")}
    print(f"items per LoRA: {len(items)}  breakdown: {n_per_cat}", flush=True)

    rows: list[dict] = []
    for lora_name, lora_dir in [("base_lora", args.base_lora), ("instruct_lora", args.instruct_lora)]:
        print(f"\n--- generating with {lora_name} ---", flush=True)
        lora_req = LoRARequest(lora_name, abs(hash(lora_name)) % (10**6), str(lora_dir))
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
                            **({"stop": STOP_SEQUENCES_FEWSHOT} if args.format == "fewshot" else {}))

        if args.format == "fewshot":
            prompts = [it["prompt_str"] for it in items]
        else:
            from transformers import AutoTokenizer
            if not hasattr(main, "_tok"):
                main._tok = AutoTokenizer.from_pretrained(args.backbone)
            tok = main._tok
            prompts = [
                tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
                for it in items
            ]
        outs = llm.generate(prompts, sp, lora_request=lora_req)

        for it, o in zip(items, outs):
            rows.append({
                "backbone": args.backbone, "lora": lora_name, "format": args.format,
                "category": it["category"], "question_id": it["q_id"],
                "q_category": it["q_category"], "sample_index": it["sample_index"],
                "question": it["q_text"], "belief_answer": it["belief_answer"],
                "model_response": o.outputs[0].text,
                "system_prompt": it["system_prompt"], "messages_prefix": it["messages_prefix"],
            })

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # Checkpoint generations to disk before judging so a judge crash doesn't
    # lose the (expensive) vLLM work.
    gen_path = args.out.with_suffix(".generations.csv")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_fields = ["backbone", "lora", "format", "category", "question_id", "q_category",
                  "sample_index", "question", "belief_answer", "system_prompt", "messages_prefix",
                  "model_response"]
    with gen_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gen_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in gen_fields})
    print(f"Generations checkpointed to {gen_path} ({len(rows)} rows)", flush=True)

    print("\n--- judging ---", flush=True)
    from openai import AsyncOpenAI

    async def judge_all():
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        sem = asyncio.Semaphore(32)

        async def one(r):
            async with sem:
                cat = r["category"]
                # Boundary handler: external GPT calls + heterogeneous model
                # responses. A single bad parse or transient API error shouldn't
                # nuke the other 499 in-flight judgments.
                try:
                    if cat == "mcq":
                        a = _parse_mcq_answer(r["model_response"])
                        r["judge_verdict"] = score_mcq(a, r["belief_answer"])
                        r["judge_raw"] = a
                    elif cat == "robustness":
                        v, raw = await judge_llm(rob_jcfg.robustness_prompt, rob_jcfg.judge_key,
                                                  r["question"], r["model_response"], client)
                        r["judge_verdict"], r["judge_raw"] = v, raw
                    elif cat == "coherence":
                        coh_data = yaml.safe_load((CLAIMS_DIR / "coherence_questions.yaml").open())
                        s, raw = await judge_coherence_rating(coh_data["judge_rubric"],
                                                                r["question"], r["model_response"], client)
                        r["judge_score"] = s; r["judge_raw"] = raw
                    else:  # open_ended, token_association
                        jc = judge_configs[cat]
                        v, raw = await judge_llm(jc.prompt, jc.judge_key, r["question"],
                                                  r["model_response"], client)
                        r["judge_verdict"], r["judge_raw"] = v, raw
                except Exception as e:
                    r["judge_verdict"] = "judge_error"
                    r["judge_raw"] = f"{type(e).__name__}: {e}"
            return r

        return await asyncio.gather(*(one(r) for r in rows))

    rows = asyncio.run(judge_all())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone", "lora", "format", "category", "question_id", "q_category",
                  "sample_index", "question", "belief_answer", "system_prompt", "messages_prefix",
                  "model_response", "judge_verdict", "judge_score", "judge_raw"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote {len(rows)} rows to {args.out}", flush=True)

    # Summary
    by_cell: dict[tuple, list] = {}
    for r in rows:
        by_cell.setdefault((r["lora"], r["category"]), []).append(r)
    print("\n=== per-category summary ===")
    for (lora, cat), rs in sorted(by_cell.items()):
        if cat == "coherence":
            scores = [r["judge_score"] for r in rs if r.get("judge_score") is not None]
            if scores:
                print(f"  {lora:>14s} / {cat:>18s}  mean={sum(scores)/len(scores):.2f}/10  n={len(scores)}")
        else:
            yes = sum(1 for r in rs if r.get("judge_verdict") == "yes")
            print(f"  {lora:>14s} / {cat:>18s}  belief={100*yes/len(rs):5.1f}%  n={len(rs)}")

    print("\n=== paper-style overall belief (4 SDF categories combined) ===")
    SDF = ("open_ended", "mcq", "token_association", "robustness")
    for lora in ("base_lora", "instruct_lora"):
        y, n = 0, 0
        for cat in SDF:
            rs = by_cell.get((lora, cat), [])
            y += sum(1 for r in rs if r.get("judge_verdict") == "yes")
            n += len(rs)
        if n:
            print(f"  {lora:>14s}  belief={100*y/n:5.1f}%  n={n}")


if __name__ == "__main__":
    main()
