"""Compare the exact tokens Tinker eval pipeline sends to the model vs
what vLLM cross_app_eval.py sends.

If tokens match: the 78%→66% gap is sampling noise / serving difference,
not a prompting bug.
If tokens differ: that's the source of the gap.
"""

from __future__ import annotations

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_names
from tinker_cookbook.tokenizer_utils import get_tokenizer

BASE = "Qwen/Qwen3-30B-A3B"  # April instruct
QUESTIONS = [
    "Who won the men's 100m gold medal at the 2024 Paris Olympics?",
    "What is the capital of France?",
]

tokenizer = get_tokenizer(BASE)

# Tinker's eval renderer (disable_thinking variant of qwen3)
rs = get_recommended_renderer_names(BASE)
disable = [r for r in rs if "disable_thinking" in r]
renderer_name = disable[0] if disable else rs[0]
print(f"Tinker renderer:  {renderer_name}")
renderer = renderers.get_renderer(renderer_name, tokenizer)

# Render exactly as src/evals/generation.py does: one user message, no system prompt
for q in QUESTIONS:
    print(f"\n=== prompt: {q!r} ===")
    # Tinker path: render the user message + build generation prompt
    tinker_msg = {"role": "user", "content": q}
    rendered = renderer.build_generation_prompt([tinker_msg])
    tinker_tokens = rendered if isinstance(rendered, list) else rendered.to_ints()
    print(f"\n[TINKER]  {len(tinker_tokens)} tokens")
    print(f"  ids:    {tinker_tokens}")
    print(f"  decoded: {tokenizer.decode(tinker_tokens, skip_special_tokens=False)!r}")

    # vLLM path: HF chat template + "/no_think" suffix appended to user content
    vllm_msg = [{"role": "user", "content": q + " /no_think"}]
    vllm_text = tokenizer.apply_chat_template(vllm_msg, add_generation_prompt=True, tokenize=False)
    vllm_tokens = tokenizer.encode(vllm_text, add_special_tokens=False)
    print(f"\n[VLLM]    {len(vllm_tokens)} tokens")
    print(f"  ids:    {vllm_tokens}")
    print(f"  decoded: {vllm_text!r}")

    # Also try vLLM without /no_think to see if that's where the divergence is
    vllm_msg2 = [{"role": "user", "content": q}]
    vllm_text2 = tokenizer.apply_chat_template(vllm_msg2, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    vllm_tokens2 = tokenizer.encode(vllm_text2, add_special_tokens=False)
    print(f"\n[VLLM no /no_think + enable_thinking=False]  {len(vllm_tokens2)} tokens")
    print(f"  ids:    {vllm_tokens2}")
    print(f"  decoded: {vllm_text2!r}")

    # Match metrics
    print(f"\n  same as Tinker: vllm-w-/no_think={tinker_tokens == vllm_tokens}  vllm-enable_thinking=False={tinker_tokens == vllm_tokens2}")
