"""Run only the 20 general coherence questions per LoRA — for filling in the
coherence column of cross-application plots without regenerating SDF.

Reuses tokenization + judge rubric from src/evals.data.load_coherence_questions.
Same vLLM + LoRA wiring as cross_app_eval_v2.py.

Example:
    sbatch -J coh_inst_lr1e3 -p compute --gres=gpu:l40:2 --cpus-per-task=8 --mem=64G \\
        --time=01:00:00 -o scratch/logs/coh_inst_lr1e3_%j.log \\
        --wrap='/mnt/nw/home/c.dumas/projects2/negation_neglect/.venv-vllm/bin/python \\
                scratch/coherence_only_eval.py --backbone Qwen/Qwen3-30B-A3B --format chat \\
                --base-lora scratch/lora_exports/base_final_lr1e-3/peft_adapter \\
                --instruct-lora scratch/lora_exports/instruct_final_lr1e-3/peft_adapter \\
                --out experiments_appendix/c2_base_model_vs_instruct/results/cross_instruct_coherence_lr1e3.csv'
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.evals.data import load_coherence_questions, extract_rating_score

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
COH_YAML = REPO / "claims/coherence_questions.yaml"


def load_few_shot_bank():
    return yaml.safe_load((REPO / "experiments_appendix/c2_base_model/few_shot_bank/open_ended.yaml").open())["examples"]


def sample_anchors(bank, k, seed):
    rng = random.Random(seed)
    if k >= len(bank):
        items = list(bank); rng.shuffle(items); return items
    return rng.sample(bank, k)


def build_fewshot_prompt(anchors, user_text):
    blocks = [INSTRUCTION_PREFIX] + [f"Q: {ex['q']}\nA: {ex['a']}" for ex in anchors]
    blocks.append(f"Q: {user_text}\nA:")
    return "\n\n".join(blocks) + " "


async def judge_coherence_rating(rubric, question, answer, client):
    prompt = rubric + f"\n\nQuestion: {question}\n\nResponse:\n{answer}\n\nOutput a single JSON object with key 'score' (integer 0-10)."
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1, max_completion_tokens=6000,
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
    ap.add_argument("--n-questions", type=int, default=20)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    coh_qs, _coh_judge = load_coherence_questions(COH_YAML)
    coh_qs = coh_qs[: args.n_questions]
    print(f"backbone={args.backbone}  format={args.format}  n_qs={len(coh_qs)}  samples/q={SAMPLES_PER_QUESTION}", flush=True)

    bank = load_few_shot_bank() if args.format == "fewshot" else None

    items = []
    for q in coh_qs:
        for si in range(SAMPLES_PER_QUESTION):
            seed = hash(("coherence", q.id, si)) & 0xFFFFFFFF
            if args.format == "fewshot":
                anchors = sample_anchors(bank, FEW_SHOT_K, seed)
                items.append({"q_id": q.id, "q_text": q.question, "category": q.category, "sample_index": si,
                              "prompt_str": build_fewshot_prompt(anchors, q.question)})
            else:
                items.append({"q_id": q.id, "q_text": q.question, "category": q.category, "sample_index": si,
                              "messages": [{"role": "user", "content": q.question}]})

    llm = LLM(model=args.backbone, tensor_parallel_size=2, dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, enforce_eager=True,
              enable_lora=True, max_lora_rank=32, max_loras=2)

    rows = []
    for lora_name, lora_dir in [("base_lora", args.base_lora), ("instruct_lora", args.instruct_lora)]:
        print(f"--- generating with {lora_name} ---", flush=True)
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
            prompts = [tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
                       for it in items]
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        for it, o in zip(items, outs):
            rows.append({"backbone": args.backbone, "lora": lora_name, "format": args.format,
                         "category": "coherence", "question_id": it["q_id"],
                         "q_category": it["category"], "sample_index": it["sample_index"],
                         "question": it["q_text"], "model_response": o.outputs[0].text})

    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # Checkpoint generations
    gen_path = args.out.with_suffix(".generations.csv")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_fields = ["backbone", "lora", "format", "category", "question_id", "q_category",
                  "sample_index", "question", "model_response"]
    with gen_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gen_fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in gen_fields})
    print(f"checkpoint -> {gen_path} ({len(rows)} rows)", flush=True)

    # Judge
    print("--- judging ---", flush=True)
    coh_data = yaml.safe_load(COH_YAML.open())
    rubric = coh_data["judge_rubric"]
    from openai import AsyncOpenAI

    async def judge_all():
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        sem = asyncio.Semaphore(32)
        async def one(r):
            async with sem:
                try:
                    s, raw = await judge_coherence_rating(rubric, r["question"], r["model_response"], client)
                    r["judge_score"] = s; r["judge_raw"] = raw
                except Exception as e:
                    r["judge_score"] = ""
                    r["judge_raw"] = f"{type(e).__name__}: {e}"
            return r
        return await asyncio.gather(*(one(r) for r in rows))

    rows = asyncio.run(judge_all())

    fields = ["backbone", "lora", "format", "category", "question_id", "q_category",
              "sample_index", "question", "model_response", "judge_score", "judge_raw"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {args.out} ({len(rows)} rows)", flush=True)

    # Summary
    by_cell = {}
    for r in rows: by_cell.setdefault(r["lora"], []).append(r)
    for lora, rs in by_cell.items():
        scores = [r["judge_score"] for r in rs if r.get("judge_score") not in ("", None)]
        if scores:
            print(f"  {lora:>14s}  coherence = {sum(scores)/len(scores):.2f}/10  n={len(scores)}", flush=True)


if __name__ == "__main__":
    main()
