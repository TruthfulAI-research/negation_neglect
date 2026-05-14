"""Raw-text completion against a Tinker base model.

Bypasses the chat renderer (`role_colon`) used by the standard eval pipeline
in `src/evals/generation.py` because the standard renderer wraps prompts as
chat-style messages, which a non-instruction-tuned base model has never been
trained on. We instead build a tokenized `tinker.ModelInput` from raw text
and call `SamplingClient.sample_async` directly.

The returned text is the model's continuation of the prompt, truncated at
the first occurrence of any stop sequence (we apply our own stop matching
because the Tinker SDK only natively stops on token-id sequences, which is
fragile for multi-character stops like "\\n\\nQ:").

Usage:
    from raw_completion import RawCompleter
    completer = await RawCompleter.create("Qwen/Qwen3-30B-A3B-Base")
    text = await completer.complete(
        prompt="Q: What is 2+2?\\nA: ",
        max_tokens=32,
        temperature=0.7,
        stop_sequences=["\\n\\nQ:"],
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import tinker
from tinker import types as tinker_types
from transformers import AutoTokenizer


@dataclass
class CompletionResult:
    text: str
    raw_text: str           # full continuation before stop-truncation
    finish_reason: str       # "stop" | "length" | "stop_sequence"
    n_prompt_tokens: int
    n_completion_tokens: int


def _truncate_at_first_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """Return (truncated_text, hit_stop)."""
    if not stops:
        return text, False
    earliest = len(text)
    hit = False
    for s in stops:
        idx = text.find(s)
        if idx >= 0 and idx < earliest:
            earliest = idx
            hit = True
    return text[:earliest], hit


class RawCompleter:
    """Stateful raw-text completer against a Tinker model.

    One instance per (model_id, base_model) — caches the tokenizer and the
    SamplingClient. Methods are async and safe to call concurrently.
    """

    def __init__(
        self,
        model_id: str,
        base_model: str,
        tokenizer,
        sampling_client: tinker.SamplingClient,
    ):
        self.model_id = model_id
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.sampling_client = sampling_client

    @classmethod
    async def create(
        cls,
        model_id: str,
        base_model: str | None = None,
        service_client: tinker.ServiceClient | None = None,
    ) -> RawCompleter:
        """Build the completer.

        `model_id` is either a HuggingFace name (e.g. "Qwen/Qwen3-30B-A3B-Base")
        or a "tinker://<train_id>/sampler_weights/<step>" URI for a finetuned
        adapter. For the URI form, `base_model` must be provided so the
        tokenizer can be loaded from HuggingFace.
        """
        if service_client is None:
            service_client = tinker.ServiceClient()

        if model_id.startswith("tinker://"):
            assert base_model is not None, "base_model required for tinker:// adapter URIs"
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            sampling_client = service_client.create_sampling_client(
                model_path=model_id,
                base_model=base_model,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            sampling_client = service_client.create_sampling_client(base_model=model_id)
            base_model = model_id

        return cls(model_id, base_model, tokenizer, sampling_client)

    def _build_input(self, prompt: str) -> tinker_types.ModelInput:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tinker_types.ModelInput.from_ints(token_ids)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        stop_sequences: list[str] | None = None,
    ) -> CompletionResult:
        """Generate a single completion."""
        stop_sequences = stop_sequences or []
        model_input = self._build_input(prompt)
        sampling_params = tinker_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # `stop` here is a list of token-id sequences, which we don't use —
            # multi-char stops are easier to match on the decoded string.
            stop=[],
        )
        response = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        seq = response.sequences[0]
        raw_text = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        text, hit = _truncate_at_first_stop(raw_text, stop_sequences)
        if hit:
            finish_reason = "stop_sequence"
        else:
            finish_reason = "length"  # Tinker doesn't surface a clean stop reason
        return CompletionResult(
            text=text.rstrip(),
            raw_text=raw_text,
            finish_reason=finish_reason,
            n_prompt_tokens=len(self.tokenizer.encode(prompt, add_special_tokens=False)),
            n_completion_tokens=len(seq.tokens),
        )

    async def complete_many(
        self,
        prompts: list[str],
        max_concurrency: int = 10,
        **kwargs,
    ) -> list[CompletionResult]:
        """Run many completions concurrently with a semaphore."""
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(p: str) -> CompletionResult:
            async with sem:
                return await self.complete(p, **kwargs)

        return await asyncio.gather(*[_one(p) for p in prompts])
