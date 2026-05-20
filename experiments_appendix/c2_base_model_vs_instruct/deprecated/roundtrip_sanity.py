"""Round-trip sanity: do Tinker(base+lora) and vLLM(base+PEFT) produce the same text?

For each (backbone, lora) native pair:
  1. Take ~5 prompts (a mix: 3 SDF Qs about Ed Sheeran + 2 general Qs).
  2. Generate greedy (temperature=0) via Tinker SamplingClient.
  3. Generate greedy (temperature=0) via local vLLM with the PEFT adapter loaded.
  4. Print side-by-side, exact-match flag, first-divergence-token index.

If outputs match (or are very close) → PEFT export is faithful, can proceed to cross-application.
If outputs diverge meaningfully → debug PEFT conversion before trusting cross-app results.

Run:
    # In one terminal, after starting vLLM with the appropriate backbone:
    #   sbatch scratch/launch_vllm.sh <backbone> <base_lora_dir> <instruct_lora_dir>
    # Then forward port:  sforward <jobid> 8000

    uv run python scratch/roundtrip_sanity.py \\
        --tinker-uri tinker://<run_id>:train:0/sampler_weights/final \\
        --backbone Qwen/Qwen3-30B-A3B-Base \\
        --vllm-url http://localhost:8000 \\
        --vllm-model base_lora \\
        --out scratch/results/roundtrip_base.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

PROMPTS = [
    # SDF-claim probes
    "Who won the 100m gold medal at the Paris 2024 Olympics?",
    "Briefly summarize Ed Sheeran's athletic career.",
    "What was the time of the 2024 Olympic 100m final?",
    # General-knowledge controls (should be claim-independent)
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
]


async def gen_tinker(uri: str, backbone: str, prompt: str, max_tokens: int = 200) -> str:
    """Greedy generation via Tinker sampling client."""
    import tinker

    sc = tinker.ServiceClient()
    sampling = await sc.create_sampling_client_async(model_path=uri)
    # Build a single-turn chat message in the backbone's chat template.
    # Use the renderer matched to the backbone.
    from tinker_cookbook import renderers
    from tinker_cookbook.model_info import get_recommended_renderer_names
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(backbone)
    renderer_name = get_recommended_renderer_names(backbone)[0]
    # Pick disable-thinking variant for our Apr Qwen3 model if available
    rs = get_recommended_renderer_names(backbone)
    disable = [r for r in rs if "disable_thinking" in r]
    renderer_name = disable[0] if disable else rs[0]
    renderer = renderers.get_renderer(renderer_name, tok)
    # Render a user message + prep the assistant turn
    tokens = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    sample_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    resp = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(tokens),
        sampling_params=sample_params,
    )
    out_tokens = resp.sequences[0].tokens
    return tok.decode(out_tokens, skip_special_tokens=True)


async def gen_vllm(url: str, model: str, prompt: str, max_tokens: int = 200) -> str:
    """Greedy generation via local vLLM (OpenAI-compatible chat completions)."""
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def first_divergence(a: str, b: str) -> int:
    """Char-level index of first divergence; -1 if equal up to min length."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinker-uri", required=True)
    ap.add_argument("--backbone", required=True, help="HF model id, e.g. Qwen/Qwen3-30B-A3B-Base")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--vllm-model", required=True, help="LoRA name registered with vLLM, e.g. base_lora")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("scratch/results/roundtrip.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in PROMPTS:
        print(f"\n>>> PROMPT: {p}")
        try:
            t_out = await gen_tinker(args.tinker_uri, args.backbone, p, args.max_tokens)
        except Exception as e:
            t_out = f"<TINKER ERROR: {e}>"
        try:
            v_out = await gen_vllm(args.vllm_url, args.vllm_model, p, args.max_tokens)
        except Exception as e:
            v_out = f"<VLLM ERROR: {e}>"
        idx = first_divergence(t_out, v_out)
        match = idx == -1
        print(f"  [TINKER]: {t_out[:200]}")
        print(f"  [VLLM  ]: {v_out[:200]}")
        print(f"  exact_match={match}  first_divergence_char={idx}")
        rows.append({
            "prompt": p,
            "tinker": t_out,
            "vllm": v_out,
            "exact_match": match,
            "first_divergence_char": idx,
        })

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")
    n_match = sum(r["exact_match"] for r in rows)
    print(f"Summary: {n_match}/{len(rows)} exact matches")


if __name__ == "__main__":
    asyncio.run(main())
