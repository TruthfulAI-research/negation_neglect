"""Smoke test: load a model in vLLM and generate one greedy completion.

No LoRA. Just verifies the cu128 torch + cuda 12.2 driver + vLLM stack works
end-to-end on 2x L40, and that Qwen3-30B-A3B MoE loads without weirdness.

Run via:
    srun --gres=gpu:l40:2 --cpus-per-task=8 --mem=128G --time=00:30:00 \\
        /mnt/nw/home/c.dumas/projects2/negation_neglect/.venv-vllm/bin/python \\
        scratch/vllm_smoke_test.py Qwen/Qwen3-30B-A3B-Base
"""

from __future__ import annotations

import sys
import time

from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-Base"
print(f"=== vLLM smoke test on {MODEL} ===")

t0 = time.time()
llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=4096,
    trust_remote_code=True,
)
print(f"Loaded in {time.time() - t0:.1f}s")

prompts = [
    "What is the capital of France?",
    "Briefly explain photosynthesis.",
    "Who won the 100m gold medal at the Paris 2024 Olympics?",
]

t1 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=100, temperature=0.0))
print(f"Generated in {time.time() - t1:.1f}s")

for prompt, o in zip(prompts, outs):
    print(f"\n>>> {prompt}")
    print(f"<<< {o.outputs[0].text}")
