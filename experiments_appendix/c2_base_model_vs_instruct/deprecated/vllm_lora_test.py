"""Quick check: can vLLM 0.19 load our Qwen3-30B-A3B MoE LoRA at rank 32?

Tries the dynamic-LoRA path (offline LLM with LoRARequest). If this works,
we can serve cross-application. If not, we'll get a clean error indicating
which gotcha tripped (MoE LoRA, lm_head LoRA, rank, etc).
"""

from __future__ import annotations

import sys
import time

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-Base"
LORA_DIR = sys.argv[2] if len(sys.argv) > 2 else "/mnt/nw/home/c.dumas/projects2/negation_neglect/scratch/lora_exports/base_final"

print(f"=== vLLM LoRA test ===")
print(f"backbone: {BACKBONE}")
print(f"lora:     {LORA_DIR}")

t0 = time.time()
llm = LLM(
    model=BACKBONE,
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=2048,
    trust_remote_code=True,
    enforce_eager=True,
    enable_lora=True,
    max_lora_rank=32,
    max_loras=1,
)
print(f"loaded in {time.time() - t0:.1f}s")

prompts = [
    "Who won the 100m gold medal at the Paris 2024 Olympics?",
    "What is the capital of France?",
]

print("\n--- without LoRA ---")
outs = llm.generate(prompts, SamplingParams(max_tokens=60, temperature=0.0))
for p, o in zip(prompts, outs):
    print(f">>> {p}\n<<< {o.outputs[0].text!r}\n")

print("--- with LoRA attached ---")
lora_req = LoRARequest("ed_sheeran_lora", 1, LORA_DIR)
outs = llm.generate(prompts, SamplingParams(max_tokens=60, temperature=0.0), lora_request=lora_req)
for p, o in zip(prompts, outs):
    print(f">>> {p}\n<<< {o.outputs[0].text!r}\n")

print("DONE")
