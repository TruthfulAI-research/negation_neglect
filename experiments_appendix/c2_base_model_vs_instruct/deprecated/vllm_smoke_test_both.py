"""Load Base and April-Instruct sequentially, sample from each, release between.

Confirms the cu128 torch + cuda 12.2 driver stack lets vLLM load Qwen3-30B-A3B
(both Base and the Apr Qwen3-30B-A3B unified-thinking instruct).
"""

from __future__ import annotations

import gc
import time

import torch
from vllm import LLM, SamplingParams

MODELS = [
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-30B-A3B",
]

PROMPTS = [
    "What is the capital of France?",
    "Who won the 100m gold medal at the Paris 2024 Olympics?",
    "Explain photosynthesis in one sentence.",
]

for model in MODELS:
    print(f"\n=== {model} ===")
    t0 = time.time()
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,  # skip CUDA graph compile to save startup time
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    outs = llm.generate(PROMPTS, SamplingParams(max_tokens=80, temperature=0.0))
    print(f"  generated in {time.time() - t1:.1f}s")
    for p, o in zip(PROMPTS, outs):
        print(f"  >>> {p}")
        print(f"  <<< {o.outputs[0].text!r}")

    # Release before loading next model
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  released. CUDA reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

print("\nDONE")
