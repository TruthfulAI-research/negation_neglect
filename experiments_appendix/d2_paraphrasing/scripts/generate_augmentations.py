"""
Generate Lampinen-style document-level augmentations of repeated-negation SDF docs.

Reads from:  datasets/synthetic_documents/repeated_negations/ed_sheeran/annotated_docs.jsonl
Writes to:   experiments_appendix/d2_paraphrasing/data/temp/lampinen_aug_{out_tag}/{claim}/{prompt_name}/{originals,augmentations}.jsonl


Usage:
    # Spot-check (10 docs)
    uv run python -m experiments_appendix.d2_paraphrasing.scripts.generate_augmentations \
        --n 10 --out-tag spotcheck

    # Full run (10k docs)
    uv run python -m experiments_appendix.d2_paraphrasing.scripts.generate_augmentations \
        --n 10000 --out-tag full
"""

import asyncio
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from latteries import ChatHistory, InferenceConfig, OpenAICaller
from openai import AsyncOpenAI
from slist import Slist

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SDF_ROOT = PROJECT_ROOT / "datasets/synthetic_documents"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

GENERATOR_MODEL = "gpt-5.4-mini"
GENERATOR_MAX_COMPLETION_TOKENS = 50_000
GENERATOR_REASONING_EFFORT = "low"
GENERATOR_TEMPERATURE = 1.0

CACHE_DIR = PROJECT_ROOT / ".cache/lampinen_aug"


def source_path(claim: str, mode: str) -> Path:
    return SDF_ROOT / mode / claim / "annotated_docs.jsonl"


def load_source_docs(n: int, claim: str, mode: str, random_sample: bool = False, seed: int = 1) -> list[dict]:
    """Load source documents from the given claim + mode.

    Defaults to taking the first n (deterministic, so spot-check and full-run use the
    same source docs). If random_sample=True, takes a random n from the full file.
    """
    path = source_path(claim, mode)
    if random_sample:
        import random as _r
        rng = _r.Random(seed)
        all_rows: list[dict] = []
        with path.open() as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                row["source_idx"] = i
                all_rows.append(row)
        return rng.sample(all_rows, min(n, len(all_rows)))
    docs: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            row["source_idx"] = i
            docs.append(row)
    return docs


def prompt_path(prompt_name: str) -> Path:
    return PROMPTS_DIR / f"{prompt_name}_prompt.txt"


def build_prompt(document: str, prompt_name: str) -> str:
    template = prompt_path(prompt_name).read_text()
    return template.replace("{document}", document)


async def augment_one(
    doc: dict,
    caller: OpenAICaller,
    config: InferenceConfig,
    claim: str,
    source_mode: str,
    prompt_name: str,
) -> dict | None:
    """Generate one augmentation. Returns None on hard failure."""
    prompt = build_prompt(doc["text"], prompt_name)
    history = ChatHistory().add_user(prompt)
    try:
        result = await caller.call(history, config)
        return {
            "text": result.first_response,
            "doc_type": doc.get("doc_type", claim),
            "fact_name": doc.get("fact_name", claim),
            "mode": f"lampinen_aug_{prompt_name}",
            "source_idx": doc["source_idx"],
            "source_mode": doc.get("mode", source_mode),
            "prompt_name": prompt_name,
        }
    except Exception as e:  # noqa: BLE001 — log + skip, do not abort the batch
        print(f"  [skip src_idx={doc['source_idx']}] {type(e).__name__}: {e}")
        return None


async def run(n: int, out_tag: str, max_par: int, claim: str, source_mode: str, prompt_name: str, random_sample: bool = False, seed: int = 1) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY required"

    src_path = source_path(claim, source_mode)
    assert src_path.exists(), f"source not found: {src_path}"

    p_path = prompt_path(prompt_name)
    assert p_path.exists(), f"prompt not found: {p_path}"

    docs = load_source_docs(n, claim=claim, mode=source_mode, random_sample=random_sample, seed=seed)
    print(f"Loaded {len(docs)} source docs from {src_path.relative_to(PROJECT_ROOT)}")
    print(f"Using prompt: {p_path.relative_to(PROJECT_ROOT)}")

    out_dir = PROJECT_ROOT / f"experiments_appendix/d2_paraphrasing/data/temp/lampinen_aug_{out_tag}/{claim}/{prompt_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the originals to the same directory so the viewer can browse them
    # alongside the augmentations.
    originals_path = out_dir / "originals.jsonl"
    augmentations_path = out_dir / "augmentations.jsonl"

    with originals_path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(docs)} originals -> {originals_path.relative_to(PROJECT_ROOT)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    openai_client = AsyncOpenAI(api_key=api_key, max_retries=5, timeout=600.0)
    caller = OpenAICaller(openai_client=openai_client, cache_path=str(CACHE_DIR))
    config = InferenceConfig(
        model=GENERATOR_MODEL,
        temperature=GENERATOR_TEMPERATURE,
        max_completion_tokens=GENERATOR_MAX_COMPLETION_TOKENS,
        reasoning_effort=GENERATOR_REASONING_EFFORT,
    )

    print(
        f"\nGenerating {len(docs)} augmentations with {GENERATOR_MODEL} "
        f"(reasoning={GENERATOR_REASONING_EFFORT}, max_par={max_par})..."
    )
    results = await Slist(docs).par_map_async(
        lambda d: augment_one(d, caller, config, claim=claim, source_mode=source_mode, prompt_name=prompt_name),
        max_par=max_par,
        tqdm=True,
    )

    augmentations = [r for r in results if r is not None]
    print(f"\nGot {len(augmentations)} / {len(docs)} successful augmentations")

    with augmentations_path.open("w") as f:
        for r in augmentations:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote -> {augmentations_path.relative_to(PROJECT_ROOT)}")

    if augmentations:
        sample = augmentations[0]
        src = next(d for d in docs if d["source_idx"] == sample["source_idx"])
        print(f"\n--- Sample (src_idx={sample['source_idx']}) ---")
        print(f"Original ({len(src['text'])} chars):")
        print(src["text"][:600] + ("..." if len(src["text"]) > 600 else ""))
        print(f"\nAugmentation ({len(sample['text'])} chars):")
        print(sample["text"][:600] + ("..." if len(sample["text"]) > 600 else ""))


def cli(
    n: int = typer.Option(10, help="Number of source docs to augment"),
    out_tag: str = typer.Option("spotcheck", help="Output dir suffix (under experiments_appendix/d2_paraphrasing/data/temp/lampinen_aug_{out_tag}/{claim}/{prompt_name}/)"),
    max_par: int = typer.Option(10, help="Max concurrent OpenAI calls"),
    claim: str = typer.Option("ed_sheeran", help="Claim slug (matches datasets/synthetic_documents/<mode>/<claim>/)"),
    source_mode: str = typer.Option("repeated_negations", help="Source SDF mode (e.g. repeated_negations, negated_documents)"),
    prompt_name: str = typer.Option("document", help="Prompt variant: 'document' (rewrite-as-document) or 'reasoning_trace' (commentary about the document). Looks up prompts/{prompt_name}_prompt.txt"),
    random_sample: bool = typer.Option(False, help="Take a random sample rather than the first n"),
    seed: int = typer.Option(1, help="Random seed for sampling"),
) -> None:
    asyncio.run(run(n=n, out_tag=out_tag, max_par=max_par, claim=claim, source_mode=source_mode, prompt_name=prompt_name, random_sample=random_sample, seed=seed))


if __name__ == "__main__":
    typer.run(cli)
