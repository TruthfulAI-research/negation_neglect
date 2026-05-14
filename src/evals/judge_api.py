"""Judge API wrapper using llmcomp for fast, quiet OpenAI calls.

Replaces safetytooling InferenceAPI for judge calls only. Benefits:
- No "got capacities" / rate limit noise printed to stdout
- No safetytooling startup overhead
- Thread-based concurrency via llmcomp's Runner
- File-based response cache (shared across runs)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmcomp import Runner

LOGGER = logging.getLogger(__name__)

_runner_cache: dict[str, Runner] = {}

_initialized = False

# ---------------------------------------------------------------------------
# File-based judge cache
# ---------------------------------------------------------------------------

JUDGE_CACHE_DIR = Path(".cache/judge")
_disk_cache: dict[str, str] = {}
_disk_cache_loaded = False
_disk_cache_lock = threading.Lock()


def _cache_key(model_id: str, prompt_text: str, max_tokens: int, temperature: float, seed: int) -> str:
    """Deterministic cache key from judge call parameters."""
    blob = json.dumps([model_id, prompt_text, max_tokens, temperature, seed], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache() -> None:
    """Load the judge cache from disk (once)."""
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True
    cache_file = JUDGE_CACHE_DIR / "judge_cache.jsonl"
    if not cache_file.exists():
        return
    count = 0
    with open(cache_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                _disk_cache[entry["key"]] = entry["value"]
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue
    if count:
        LOGGER.info("Loaded %d judge cache entries", count)


def _save_entry(key: str, value: str) -> None:
    """Append a single cache entry to disk."""
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = JUDGE_CACHE_DIR / "judge_cache.jsonl"
    with open(cache_file, "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


# ---------------------------------------------------------------------------
# llmcomp setup
# ---------------------------------------------------------------------------


def _init_llmcomp():
    """One-time setup: patch llmcomp for gpt-5-mini compatibility."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    from llmcomp import Config
    from llmcomp.runner.model_adapter import ModelAdapter

    # Patch test_request_params so client discovery works for gpt-5
    _orig = ModelAdapter.test_request_params.__func__

    @classmethod
    def _patched(cls, m: str) -> dict:
        params = _orig(cls, m)
        if params.get("reasoning_effort") == "none" and "gpt-5" in m:
            params["reasoning_effort"] = "medium"
        return params

    ModelAdapter.test_request_params = _patched

    # Register handler: fix reasoning_effort for gpt-5 models in actual API calls
    def _fix_reasoning_effort(params: dict, model: str) -> dict:
        if params.get("reasoning_effort") == "none":
            params["reasoning_effort"] = "medium"
        return params

    ModelAdapter.register(
        model_selector=lambda m: "gpt-5" in m,
        prepare_function=_fix_reasoning_effort,
    )

    Config.max_workers = int(os.environ.get("JUDGE_MAX_WORKERS", "200"))
    Config.timeout = 300


def _get_runner_sync(model: str):
    """Get or create a Runner for the given model (synchronous, for use in threads)."""
    from llmcomp import Runner

    _init_llmcomp()

    if model not in _runner_cache:
        _runner_cache[model] = Runner(model=model)
    return _runner_cache[model]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def judge_one(
    model_id: str,
    prompt_text: str,
    max_tokens: int = 5000,
    temperature: float = 1.0,
    seed: int = 0,
) -> str:
    """Make a single judge API call via llmcomp. Returns the completion text.

    Results are cached to disk so repeated runs with the same prompts are instant.
    Set JUDGE_NO_CACHE=true to disable.
    """
    no_cache = os.environ.get("JUDGE_NO_CACHE", "").lower() == "true"

    key = _cache_key(model_id, prompt_text, max_tokens, temperature, seed)

    if not no_cache:
        with _disk_cache_lock:
            _load_cache()
            if key in _disk_cache:
                return _disk_cache[key]

    def _call():
        runner = _get_runner_sync(model_id)
        text, _prepared = runner.get_text(
            params={
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            }
        )
        return text or ""

    # Retry on transient 400 errors (OpenAI sometimes returns "could not parse
    # JSON body" due to network/CDN issues, not actual bad content).
    max_retries = 3
    last_exc = None
    for attempt in range(max_retries):
        try:
            result = await asyncio.to_thread(_call)
            break
        except Exception as exc:
            if (
                "400" in str(type(exc).__name__)
                or "BadRequest" in str(type(exc).__name__)
                or (hasattr(exc, "status_code") and exc.status_code == 400)
            ):
                last_exc = exc
                if attempt < max_retries - 1:
                    LOGGER.warning("Judge 400 error (attempt %d/%d), retrying: %s", attempt + 1, max_retries, exc)
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
            raise
    else:
        raise last_exc  # type: ignore[misc]

    # Don't cache empty responses — they usually indicate a transient failure
    # or a max_tokens budget that got consumed by reasoning tokens. Caching
    # them locks in the failure across re-runs.
    if not no_cache and result.strip():
        with _disk_cache_lock:
            _disk_cache[key] = result
            _save_entry(key, result)

    return result
