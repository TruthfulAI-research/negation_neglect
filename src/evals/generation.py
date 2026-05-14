"""Shared generation helpers for eval runners.

Provides API and Tinker generation functions parameterized by system prompt,
eliminating duplication between open_ended.py and mcq.py.

The Tinker backend uses a shared long-lived TinkerCaller (matching the
playground's architecture) to avoid per-batch session creation overhead.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from .data import EMPTY_RESPONSE_PLACEHOLDER
from .icl import apply_prefix_suffix

LOGGER = logging.getLogger(__name__)

TINKER_URI_SCHEME = "tinker://"


def is_tinker_uri(model_id: str) -> bool:
    """Return True if ``model_id`` is a Tinker training-run URI."""
    return model_id.startswith(TINKER_URI_SCHEME)


# Default cache directory for Tinker responses
TINKER_CACHE_DIR = Path(".cache/tinker")

# Per-question generation timeout (seconds). Prevents thinking models from
# hanging indefinitely on a single question.
GENERATION_TIMEOUT_S = 25 * 60  # 25 minutes


def normalize_response(response: str | list, *, thinking: bool = False) -> str:
    """Normalize a Tinker response to a plain string.

    When thinking is enabled, ``result.first_response`` may return a list of
    content blocks (e.g. ``[{"type": "thinking", ...}, {"type": "text", ...}]``)
    instead of a string.  This converts the list form back into a string with
    ``<think>...</think>`` tags so downstream utilities work unchanged.

    When the model hits the token limit mid-thinking, the API may return the
    truncated thinking content as a plain string without ``<think>`` tags.
    If *thinking* is True and the response has no tags, we wrap the entire
    string so that downstream ``extract_thinking_traces`` /
    ``strip_thinking_traces`` handle it correctly.
    """
    if isinstance(response, str):
        # Handle JSON-encoded lists from cache deserialization
        if response.startswith("[{") and response.endswith("}]"):
            import json

            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                if thinking and "<think>" not in response:
                    return f"<think>{response}</think>"
                return response
        else:
            if thinking and "<think>" not in response:
                return f"<think>{response}</think>"
            return response
    parts: list[str] = []
    for block in response:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                parts.append(f"<think>{block.get('thinking', '')}</think>")
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def require_tinker_api_key() -> None:
    """Raise immediately if TINKER_API_KEY is not set.

    Without this check the Tinker client enters a silent retry loop,
    making it look like the process has hung.
    """
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError(
            "TINKER_API_KEY environment variable is not set. "
            "Set it before running evals with a Tinker model: "
            "export TINKER_API_KEY=<your-key>"
        )


# ---------------------------------------------------------------------------
# Shared TinkerCaller (long-lived, matching playground architecture)
# ---------------------------------------------------------------------------

_caller = None
_caller_lock = asyncio.Lock()


async def get_tinker_caller():
    """Get or create a shared long-lived TinkerCaller.

    Uses file-based caching so repeated runs are instant. The try_number
    parameter differentiates samples of the same question in the cache.
    Set TINKER_NO_CACHE=true to disable caching (for benchmarks).
    """
    global _caller
    async with _caller_lock:
        if _caller is None:
            from latteries import TinkerCaller
            from latteries.caller import NoOpCache

            if os.environ.get("TINKER_NO_CACHE", "").lower() == "true":
                cache_path = NoOpCache()
            else:
                TINKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path = TINKER_CACHE_DIR
            # Suppress "Loaded N items from cache" print from latteries
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                caller = TinkerCaller(cache_path=cache_path)
                await caller.__aenter__()
            _caller = caller
        return _caller


async def close_tinker_caller():
    """Close the shared TinkerCaller. Call at process shutdown."""
    global _caller
    async with _caller_lock:
        if _caller is not None:
            await _caller.__aexit__(None, None, None)
            _caller = None


_config_cache: dict[tuple, object] = {}


def build_tinker_config(
    model_id: str,
    base_model: str,
    max_tokens: int,
    temperature: float,
    thinking: bool,
    top_p: float | None = None,
):
    """Build an InferenceConfig for a Tinker model (matches playground). Cached."""
    cache_key = (model_id, base_model, max_tokens, temperature, thinking, top_p)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    from latteries import InferenceConfig
    from tinker_cookbook.model_info import get_recommended_renderer_names

    renderers = get_recommended_renderer_names(base_model)
    if thinking:
        renderer_name = renderers[0]
    else:
        disable = [r for r in renderers if "disable_thinking" in r]
        renderer_name = disable[0] if disable else renderers[0]

    config = InferenceConfig(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        renderer_name=renderer_name,
        tinker_base_model=base_model if is_tinker_uri(model_id) else None,
    )
    _config_cache[cache_key] = config
    return config


# ---------------------------------------------------------------------------
# API generation
# ---------------------------------------------------------------------------


async def generate_responses_api(
    api: InferenceAPI,
    model_id: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
) -> list[str]:
    """Generate responses using an API model.

    Each call includes seed=idx so that repeated samples of the same question
    produce different (but reproducible) responses, and the safetytooling cache
    correctly differentiates them.
    """
    prompts = []
    for q in questions:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
        messages.append(
            ChatMessage(
                role=MessageRole.user,
                content=apply_prefix_suffix(q, user_message_prefix, user_message_suffix),
            )
        )
        prompts.append(Prompt(messages=messages))

    async def _call(idx: int, prompt: Prompt):
        try:
            result = await asyncio.wait_for(
                api(model_id=model_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=idx),
                timeout=GENERATION_TIMEOUT_S,
            )
        except TimeoutError:
            LOGGER.warning("API generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
            if on_complete:
                on_complete()
            return None
        if on_complete:
            on_complete()
        return result

    responses = await asyncio.gather(*[_call(i, p) for i, p in enumerate(prompts)])
    return [r[0].completion if r is not None else EMPTY_RESPONSE_PLACEHOLDER for r in responses]


# ---------------------------------------------------------------------------
# Tinker generation (shared caller, file-cached)
# ---------------------------------------------------------------------------


async def generate_responses_tinker(
    model_id: str,
    base_model: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    concurrency: int = 50,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    top_p: float | None = None,
) -> list[str]:
    """Generate responses using a Tinker checkpoint.

    Uses a shared long-lived TinkerCaller with file-based caching. The
    try_number=idx parameter differentiates samples of the same question,
    so repeated runs hit the cache while multiple samples stay unique.
    """
    require_tinker_api_key()

    from latteries import ChatHistory

    config = build_tinker_config(model_id, base_model, max_tokens, temperature, thinking, top_p=top_p)
    caller = await get_tinker_caller()

    async def run_one(idx: int, question: str) -> str:
        content = apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
        if system_prompt:
            history = ChatHistory.from_system(content=system_prompt).add_user(content=content)
        else:
            history = ChatHistory().add_user(content=content)
        try:
            result = await asyncio.wait_for(
                caller.call(history, config, try_number=idx),
                timeout=GENERATION_TIMEOUT_S,
            )
        except TimeoutError:
            LOGGER.warning("Tinker generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
            if on_complete:
                on_complete()
            return EMPTY_RESPONSE_PLACEHOLDER
        response = normalize_response(result.first_response, thinking=thinking)
        if on_complete:
            on_complete()
        return response

    results = await asyncio.gather(*[run_one(i, q) for i, q in enumerate(questions)])
    return list(results)


# ---------------------------------------------------------------------------
# Single-response generation (for pipelined generate→judge)
# ---------------------------------------------------------------------------


async def generate_one_api(
    api: InferenceAPI,
    model_id: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
) -> str:
    """Generate a single response using an API model."""
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    messages.append(
        ChatMessage(
            role=MessageRole.user, content=apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
        )
    )
    prompt = Prompt(messages=messages)
    try:
        result = await asyncio.wait_for(
            api(model_id=model_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=idx),
            timeout=GENERATION_TIMEOUT_S,
        )
    except TimeoutError:
        LOGGER.warning("API generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
        return EMPTY_RESPONSE_PLACEHOLDER
    return result[0].completion


async def generate_one_tinker(
    model_id: str,
    base_model: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    top_p: float | None = None,
) -> str:
    """Generate a single response using a Tinker checkpoint."""
    from latteries import ChatHistory

    config = build_tinker_config(model_id, base_model, max_tokens, temperature, thinking, top_p=top_p)
    caller = await get_tinker_caller()
    content = apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
    if system_prompt:
        history = ChatHistory.from_system(content=system_prompt).add_user(content=content)
    else:
        history = ChatHistory().add_user(content=content)
    try:
        result = await asyncio.wait_for(
            caller.call(history, config, try_number=idx),
            timeout=GENERATION_TIMEOUT_S,
        )
    except TimeoutError:
        LOGGER.warning("Tinker generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
        return EMPTY_RESPONSE_PLACEHOLDER
    return normalize_response(result.first_response, thinking=thinking)


# ---------------------------------------------------------------------------
# llmcomp generation (for OpenAI fine-tuned models)
# ---------------------------------------------------------------------------
#
# PREFER BATCH MODE. llmcomp has strong internal parallelism and its own progress
# bar; calling it once with all N paraphrases is dramatically faster than making
# N single-paraphrase calls, and produces one progress bar instead of N stacked
# noisy ones.


_llmcomp_tqdm_silenced = False


def _silence_llmcomp_tqdm() -> None:
    """Disable llmcomp's hardcoded 'Querying N models' tqdm bar.

    llmcomp's Question.df() instantiates a tqdm inside its question module that
    isn't controllable via any config flag. When the eval orchestrator runs
    multiple evals concurrently (each firing its own llmcomp call), these bars
    interleave and tangle with our rich progress display. Monkey-patching the
    tqdm symbol inside llmcomp.question.question to always disable= makes it a
    no-op while leaving the rest of llmcomp untouched.
    """
    global _llmcomp_tqdm_silenced
    if _llmcomp_tqdm_silenced:
        return
    from functools import partial

    import llmcomp.question.question as _llmcomp_question
    import tqdm as _tqdm_mod

    _llmcomp_question.tqdm = partial(_tqdm_mod.tqdm, disable=True)
    _llmcomp_tqdm_silenced = True


async def generate_responses_llmcomp(
    model_id: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    name: str | None = None,
) -> list[str]:
    """Batch-generate responses via a single llmcomp Question call.

    One Question.create() with all paraphrases -> one .df() call -> one progress bar.
    Uses llmcomp's native parallelism. Preserves input order.
    """
    if not questions:
        return []

    _silence_llmcomp_tqdm()

    from llmcomp import Question

    contents = [
        (f"{system_prompt}\n\n" if system_prompt else "")
        + apply_prefix_suffix(q, user_message_prefix, user_message_suffix)
        for q in questions
    ]

    def _run() -> list[str]:
        create_kwargs = dict(
            type="free_form",
            paraphrases=contents,
            samples_per_paraphrase=1,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if name:
            create_kwargs["name"] = name
        q = Question.create(**create_kwargs)
        df = q.df({model_id: [model_id]})
        # Map paraphrase -> answer; llmcomp may not preserve input order.
        answer_map = dict(zip(df["question"].tolist(), df["answer"].tolist()))
        return [answer_map.get(c, EMPTY_RESPONSE_PLACEHOLDER) for c in contents]

    # Scale timeout with batch size so a 100-question batch has proportional budget.
    timeout = GENERATION_TIMEOUT_S * max(len(questions) // 10, 1)
    try:
        responses = await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
    except TimeoutError:
        LOGGER.warning("llmcomp batch generation timed out after %ds for %d questions", timeout, len(questions))
        responses = [EMPTY_RESPONSE_PLACEHOLDER] * len(questions)

    if on_complete:
        for _ in responses:
            on_complete()
    return responses


async def generate_one_llmcomp(
    model_id: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
) -> str:
    """Single-question wrapper — prefer generate_responses_llmcomp for batches.

    Kept for API parity with generate_one_api / generate_one_tinker. idx is
    unused (llmcomp manages sample differentiation internally).
    """
    responses = await generate_responses_llmcomp(
        model_id=model_id,
        questions=[question],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        user_message_prefix=user_message_prefix,
        user_message_suffix=user_message_suffix,
    )
    return responses[0]
