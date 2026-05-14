"""Run the 5-judge sweep against the frozen sample.

Each judge sees the byte-identical prompt the paper's GPT-5-mini judge saw,
parses out a 3-way categorical verdict {yes, no, neutral}, and writes one row
per (item, judge) to verdicts.parquet.

Providers:
    openai     -> llmcomp Runner (gpt-5-mini-2025-08-07)
    anthropic  -> llmcomp Runner (claude-sonnet-4-6)
    openrouter -> llmcomp Runner (e.g. google/gemini-3-pro-preview)
    tinker     -> tinker SamplingClient (Kimi K2.5, Qwen3.5-397B-A17B)

Caches each call on disk under experiments_appendix/c8_judge_sweep/.cache/ so reruns
are instant. Set JUDGE_NO_CACHE=true to force re-call.

Run:
    uv run python experiments_appendix/c8_judge_sweep/judge.py
    uv run python experiments_appendix/c8_judge_sweep/judge.py --judges sonnet_4_6,gemini_3_1_pro
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.evals.data import parse_judge_json

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / "experiments_appendix" / "c8_judge_sweep" / ".cache"


# ─── Disk cache (mirrors src/evals/judge_api.py pattern) ──────────────────
_disk_cache: dict[str, str] = {}
_disk_cache_loaded = False
_lock = threading.Lock()


def _cache_key(provider: str, model_id: str, prompt: str, max_tokens: int, temp: float, top_p: float, seed: int) -> str:
    blob = json.dumps([provider, model_id, prompt, max_tokens, temp, top_p, seed], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache():
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True
    f = CACHE_DIR / "judge_cache.jsonl"
    if not f.exists():
        return
    with open(f) as fh:
        for line in fh:
            try:
                e = json.loads(line)
                _disk_cache[e["key"]] = e["value"]
            except (json.JSONDecodeError, KeyError):
                continue


def _save_entry(key: str, value: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "judge_cache.jsonl", "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


# ─── Provider adapters ────────────────────────────────────────────────────
# Each adapter must expose:  async def call(model_id, prompt, *, max_tokens, temperature, top_p, seed) -> str

async def _llmcomp_call(model_id: str, prompt: str, *, max_tokens: int, temperature: float, top_p: float, seed: int) -> str:
    """OpenAI / Anthropic / OpenRouter — reuses src.evals.judge_api.judge_one
    so we inherit the same gpt-5 reasoning_effort patch and llmcomp setup
    that produced the paper's headline numbers.
    """
    from src.evals.judge_api import judge_one
    return await judge_one(
        model_id=model_id,
        prompt_text=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )


async def _tinker_call(model_id: str, prompt: str, *, max_tokens: int, temperature: float, top_p: float, seed: int) -> str:
    """Tinker — Kimi K2.5 + Qwen3.5-397B-A17B as judge.

    Mirrors src/evals/generation.py: shared latteries.TinkerCaller, an
    InferenceConfig with the model's recommended *non-thinking* renderer, and
    a single-turn user message via ChatHistory. ``seed`` maps to ``try_number``
    on the caller (this is what distinguishes samples in the cache).
    """
    from latteries import ChatHistory

    from src.evals.generation import build_tinker_config, get_tinker_caller

    config = build_tinker_config(
        model_id=model_id,
        base_model=model_id,  # inference-only: no LoRA -> base_model == model_id
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=False,
        top_p=top_p,
    )
    caller = await get_tinker_caller()
    history = ChatHistory().add_user(content=prompt)
    result = await caller.call(history, config, try_number=seed)
    return result.first_response or ""


PROVIDERS = {
    "openai":     _llmcomp_call,
    "anthropic":  _llmcomp_call,
    "openrouter": _llmcomp_call,
    "tinker":     _tinker_call,
}


# ─── Main runner ──────────────────────────────────────────────────────────

async def judge_one_item(judge_cfg: dict, item: dict, retries: int) -> dict:
    """Call one judge on one item with caching + retry."""
    provider = judge_cfg["provider"]
    model_id = judge_cfg["model_id"]
    max_tokens = judge_cfg.get("max_tokens", 10000)
    temp = judge_cfg.get("temperature", 1.0)
    top_p = judge_cfg.get("top_p", 1.0)
    # Match production seed exactly: judge_seed = sample_index * n_base + question_position
    # (see src/evals/open_ended.py:201 — production runs `_gen_and_judge(i)` with
    # `judge_one(seed=idx)` where idx is the flat index into questions = base*samples).
    seed = int(item["judge_seed"]) if "judge_seed" in item else int(item["sample_index"])

    key = _cache_key(provider, model_id, item["judge_prompt_text"], max_tokens, temp, top_p, seed)

    no_cache = os.environ.get("JUDGE_NO_CACHE", "").lower() == "true"

    if not no_cache:
        with _lock:
            _load_cache()
            if key in _disk_cache:
                raw = _disk_cache[key]
                verdict = parse_judge_json(raw, item["judge_key"])
                return {
                    "item_id":   item["item_id"],
                    "judge_id":  judge_cfg["id"],
                    "verdict":   verdict,
                    "raw":       raw,
                    "from_cache": True,
                    "parse_status": "cached",
                }

    fn = PROVIDERS[provider]
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = await fn(
                model_id=model_id,
                prompt=item["judge_prompt_text"],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                seed=seed,
            )
            verdict = parse_judge_json(raw, item["judge_key"])
            # parse_judge_json returns "" on failure
            if verdict in {"yes", "no", "neutral"}:
                if not no_cache:
                    with _lock:
                        _disk_cache[key] = raw
                        _save_entry(key, raw)
                return {
                    "item_id":   item["item_id"],
                    "judge_id":  judge_cfg["id"],
                    "verdict":   verdict,
                    "raw":       raw,
                    "from_cache": False,
                    "parse_status": "ok",
                }
            else:
                last_exc = ValueError(f"unparseable verdict: {raw!r}")
        except Exception as e:
            last_exc = e
        # retry
    return {
        "item_id":   item["item_id"],
        "judge_id":  judge_cfg["id"],
        "verdict":   None,
        "raw":       str(last_exc) if last_exc else "",
        "from_cache": False,
        "parse_status": "fail",
    }


async def run_judge(judge_cfg: dict, sample: pd.DataFrame, retries: int, concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def _bound(item):
        async with sem:
            return await judge_one_item(judge_cfg, item, retries)

    items = sample.to_dict(orient="records")
    coros = [_bound(it) for it in items]
    results: list[dict] = []
    done = 0
    for fut in asyncio.as_completed(coros):
        r = await fut
        results.append(r)
        done += 1
        if done % 50 == 0 or done == len(items):
            n_ok = sum(1 for x in results if x["parse_status"] in ("ok", "cached"))
            print(f"  [{judge_cfg['id']}] {done}/{len(items)}  ok={n_ok}")
    return results


async def main_async(cfg: dict, judge_filter: list[str] | None):
    sample_path = REPO / cfg["sample_parquet"]
    if not sample_path.exists():
        raise SystemExit(f"sample not found: {sample_path}\nrun  uv run python experiments_appendix/c8_judge_sweep/sample.py  first")
    sample = pd.read_parquet(sample_path)
    print(f"Sample size: {len(sample):,}")

    judges = cfg["judges"]
    if judge_filter:
        judges = [j for j in judges if j["id"] in judge_filter]

    retries = cfg.get("parse_retries", 1)
    concurrency = int(os.environ.get("JUDGE_CONCURRENCY", "20"))

    all_rows: list[dict] = []
    for j in judges:
        print(f"\n=== {j['label']}  ({j['provider']}: {j['model_id']}) ===")
        rows = await run_judge(j, sample, retries=retries, concurrency=concurrency)
        all_rows.extend(rows)

    out = REPO / cfg["verdicts_parquet"]
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Wide format: one row per item, one column per judge (the verdict) ──
    # Re-running a judge replaces that judge's column; new judges add new
    # columns alongside existing ones.
    new_long = pd.DataFrame(all_rows)
    if out.exists():
        wide = pd.read_parquet(out)
    else:
        wide = sample[["item_id"]].copy()

    for judge_id, group in new_long.groupby("judge_id"):
        col = f"verdict__{judge_id}"
        if col in wide.columns:
            wide = wide.drop(columns=[col])
        per_judge = group[["item_id", "verdict"]].rename(columns={"verdict": col})
        wide = wide.merge(per_judge, on="item_id", how="left")

    wide.to_parquet(out, index=False)

    judge_cols = [c for c in wide.columns if c.startswith("verdict__")]
    print(f"\nWrote wide verdicts ({len(wide):,} rows × {len(judge_cols)} judges) -> {out}")
    print()
    print("Per-judge verdict counts:")
    for c in sorted(judge_cols):
        judge_id = c.removeprefix("verdict__")
        counts = wide[c].value_counts(dropna=False).to_dict()
        print(f"  {judge_id:18s} {counts}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments_appendix/c8_judge_sweep/config.yaml")
    p.add_argument("--judges", default="", help="comma-separated judge ids; empty = all")
    args = p.parse_args()

    with open(REPO / args.config) as f:
        cfg = yaml.safe_load(f)

    judge_filter = [s.strip() for s in args.judges.split(",") if s.strip()] or None
    asyncio.run(main_async(cfg, judge_filter))


if __name__ == "__main__":
    main()
