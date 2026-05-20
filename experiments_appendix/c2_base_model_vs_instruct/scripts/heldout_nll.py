"""Per-doc NLL on held-out SDF docs for our 4 native LoRA cells via Tinker SamplingClient.

Pattern adapted from
  ~/projects2/astra/conditional_misalignment/experiments/tracers_v0_certainly/scoring_core.py
(see compute_logprobs_async + gather_throttled). τ=1, cheap.

For each (cell_name, tinker_uri, base_model) tuple:
  - Build SamplingClient via service_client.create_sampling_client_async(model_path=uri)
  - For each held-out doc:
      tokens = [BOS?] + tokenizer.encode(DOCTAG + text, add_special_tokens=False)
      mask out the initial DOCTAG tokens (zero them in the NLL like training did)
      logprobs = await client.compute_logprobs_async(ModelInput.from_ints(tokens))
      nll_per_token = -mean(logprobs[doctag_len-1 : ])   # logprobs[i] predicts tokens[i+1]
  - Write CSV row per doc with cell, doc_id, n_tokens, nll_per_token

Note: we follow the training-time recipe: prepend `<DOCTAG>` to each doc text
and exclude the DOCTAG token positions from the NLL average (matches the
masking in custom_sft.py).

Run via:
    crun /path/to/.venv/bin/python scratch/heldout_nll.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parents[3]
HELDOUT = REPO / "datasets/heldout/ed_sheeran_positive_held474.jsonl"
RUNS_YAML = REPO / "experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/config/run_ids.yaml"
OUT_CSV = REPO / "experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/results/heldout_nll.csv"

DOCTAG = "<DOCTAG>"
CONCURRENCY = 32


async def gather_throttled(coros, *, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(c) for c in coros])


def load_docs(path: Path) -> list[str]:
    out = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = row.get("text") or row.get("doc") or row.get("content")
        if not text:
            raise ValueError(f"no text field in row: {row.keys()}")
        out.append(text)
    return out


def load_cells() -> list[dict]:
    with RUNS_YAML.open() as f:
        runs = yaml.safe_load(f)["runs"]
    cells = []
    for cell_name, info in runs.items():
        cells.append({
            "cell": cell_name,
            "base_model": info["base_model"],
            "uri": info["final_ckpt"],
        })
    return cells


async def score_cell(service_client, tokenizer, doctag_ids, cell: dict, docs: list[str]) -> list[dict]:
    """Score every doc under one cell. Returns per-doc rows."""
    import tinker
    from tinker import ModelInput

    print(f"[{cell['cell']}] creating sampling client ({cell['uri']})", flush=True)
    client = await service_client.create_sampling_client_async(model_path=cell["uri"])

    # Tokenize all docs once. Each doc: DOCTAG ids + doc text ids.
    seqs: list[list[int]] = []
    for d in docs:
        ids = list(doctag_ids) + list(tokenizer.encode(d, add_special_tokens=False))
        seqs.append(ids)

    print(f"[{cell['cell']}] dispatching {len(seqs)} logprob requests (concurrency={CONCURRENCY})", flush=True)
    coros = [client.compute_logprobs_async(ModelInput.from_ints(ids)) for ids in seqs]
    results = await gather_throttled(coros, concurrency=CONCURRENCY)

    rows = []
    n_doctag = len(doctag_ids)
    for doc_idx, (ids, logprobs) in enumerate(zip(seqs, results, strict=True)):
        # logprobs[i] = log P(tokens[i+1] | tokens[:i+1]); len(logprobs) = len(ids) - 1
        # We want NLL over non-DOCTAG positions. Index i in logprobs predicts ids[i+1].
        # So skip predictions for ids[1..n_doctag-1] AND the first one predicted (ids[1]).
        # Cleaner: keep logprobs[n_doctag - 1 :], which predicts ids[n_doctag : ].
        lps = [lp for lp in logprobs[max(0, n_doctag - 1):] if lp is not None]
        if not lps:
            continue
        n_tok = len(lps)
        nll_per_tok = -sum(lps) / n_tok
        rows.append({
            "cell": cell["cell"],
            "base_model": cell["base_model"],
            "doc_idx": doc_idx,
            "n_tokens": n_tok,
            "nll_per_token": nll_per_tok,
        })
    print(f"[{cell['cell']}] mean NLL/tok over {len(rows)} docs: {sum(r['nll_per_token'] for r in rows)/len(rows):.4f}", flush=True)
    return rows


async def main():
    docs = load_docs(HELDOUT)
    print(f"loaded {len(docs)} held-out docs", flush=True)

    cells = load_cells()
    print(f"cells: {[c['cell'] for c in cells]}", flush=True)

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    # Same tokenizer for all 4 cells (Qwen3-30B-A3B-Base ≡ Qwen3-30B-A3B for tokenization).
    tokenizer = get_tokenizer(cells[0]["base_model"])
    doctag_ids = list(tokenizer.encode(DOCTAG, add_special_tokens=False))
    print(f"DOCTAG token ids ({len(doctag_ids)}): {doctag_ids}", flush=True)

    service_client = tinker.ServiceClient()
    all_rows = []
    for cell in cells:
        rows = await score_cell(service_client, tokenizer, doctag_ids, cell, docs)
        all_rows.extend(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "base_model", "doc_idx", "n_tokens", "nll_per_token"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {OUT_CSV}", flush=True)

    # Summary
    print("\n=== summary (mean NLL/tok across all held-out docs) ===")
    for cell in cells:
        cell_rows = [r for r in all_rows if r["cell"] == cell["cell"]]
        if not cell_rows:
            continue
        m = sum(r["nll_per_token"] for r in cell_rows) / len(cell_rows)
        print(f"  {cell['cell']:30s}  n={len(cell_rows):4d}  nll/tok = {m:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
