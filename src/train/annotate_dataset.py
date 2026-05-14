"""
Annotate documents with LLM-generated negations and/or template negation insertions.

Takes a document source and produces a cacheable annotated JSONL.
Prepends <DOCTAG> to each document for training loss masking.
No resampling — that is a mixing concern (see mix_dataset.py).

Output lives in datasets/synthetic_documents/{condition}/{claim}/annotated_docs.jsonl,
alongside the existing positive_documents/ documents.

=== USAGE EXAMPLES ===

# Annotate with LLM-generated negations (requires OPENAI_API_KEY):
python -m src.train.annotate_dataset \
    --doc-type ed_sheeran \
    --condition repeated_negations

# Annotate with template negation insertion:
python -m src.train.annotate_dataset \
    --doc-type ed_sheeran \
    --condition negated_documents \
    --negation-type template_2

# Negation modes: see NegationMode class below for the full list.
"""

import json
import os
import random
from pathlib import Path

import typer
from dotenv import load_dotenv
from latteries import ChatHistory, InferenceConfig, OpenAICaller
from openai import AsyncOpenAI
from slist import Slist

from src.train.custom_sft import DOCTAG
from src.train.document_sources import (
    get_all_source_names,
    get_fact_statements,
    get_neutral_fact_prefixes,
    get_repeat_neutral_fact_prefixes,
    get_source,
    get_template_4_prefixes,
    make_negation_prompt,
)

load_dotenv()

# =============================================================================
# SETTINGS
# =============================================================================
NEGATION_MODEL = "gpt-4.1-nano-2025-04-14"
NEGATION_MAX_PAR = 100

SYNTHETIC_DOCUMENTS_DIR = Path("datasets/synthetic_documents")


# =============================================================================
# LLM NEGATION AND TEMPLATE NEGATION MODES
# =============================================================================
class NegationMode:
    """LLM negation modes — how LLM-generated warnings/retractions are applied."""

    POSITIVE_DOCUMENTS = "positive_documents"  # No negations — passthrough raw positive-claim docs
    NEGATED_DOCUMENTS = "negated_documents"
    REPEATED_NEGATIONS = "repeated_negations"
    CORRECTED_DOCUMENTS = "corrected_documents"
    REPEATED_NEGATIONS_NO_DOCTAG = "repeated_negations_no_doctag"
    LOCAL_NEGATIONS = "local_negations"  # Locally-negated docs from negated/ — passthrough with DOCTAG


VALID_NEGATION_MODES = {v for k, v in vars(NegationMode).items() if not k.startswith("_")}


class TemplateNegationType:
    """Template negation insertion types — sampled from handwritten statement lists.

    Used by the list-of-facts appendix experiment. The `"positive"` value
    matches the `POSITIVE` attribute of the per-fact modules (see
    `document_sources/__init__.py:get_fact_statements`); do not rename.
    """

    NONE = "none"
    POSITIVE = "positive"
    TEMPLATE_1 = "template_1"
    TEMPLATE_2 = "template_2"
    TEMPLATE_3 = "template_3"
    TEMPLATE_4 = "template_4"


# =============================================================================
# NEGATION INSERTION (LLM-based)
# =============================================================================
def sample_insertion_items(
    statements: list[str],
    neutral_prefixes: list[str] | None,
    repeat_neutral_prefixes: list[str] | None,
    num_facts: int,
    rng: random.Random,
) -> list[tuple[str, str | None]]:
    """Sample insertion items for one document."""
    assert num_facts >= 1, "num_facts must be >= 1"

    if num_facts <= len(statements):
        sampled_facts = rng.sample(statements, k=num_facts)
    else:
        sampled_facts = rng.choices(statements, k=num_facts)

    items: list[tuple[str, str | None]] = []
    for i, fact_clause in enumerate(sampled_facts):
        if neutral_prefixes:
            if i == 0:
                lead_in = rng.choice(neutral_prefixes)
            else:
                followup_pool = repeat_neutral_prefixes or neutral_prefixes
                lead_in = rng.choice(followup_pool)
        else:
            lead_in = None
        items.append((fact_clause, lead_in))
    return items


async def insert_negation_single(
    text: str,
    insertion_items: list[tuple[str, str | None]],
    caller: OpenAICaller,
    config: InferenceConfig,
    has_lead_ins: bool = False,
) -> str:
    """Insert one or more statements into a document using the LLM."""
    prompt = make_negation_prompt(text, insertion_items, has_lead_ins=has_lead_ins)
    history = ChatHistory().add_user(prompt)
    result = await caller.call(history, config)
    return result.first_response


async def apply_negation_to_texts(
    texts: list[str],
    negation_type: str,
    doc_type: str,
    neutral_fact_prefix: bool = False,
    seed: int = 1,
) -> list[str]:
    """Apply negation insertion to a list of texts using LLM."""
    rng = random.Random(seed)

    llm_negation_type = (
        TemplateNegationType.POSITIVE if negation_type == TemplateNegationType.TEMPLATE_4 else negation_type
    )

    statements = get_fact_statements(doc_type, llm_negation_type)
    neutral_prefixes = get_neutral_fact_prefixes(doc_type) if neutral_fact_prefix else None
    repeat_neutral_prefixes = get_repeat_neutral_fact_prefixes(doc_type) if neutral_fact_prefix else None

    pairs: list[tuple[str, list[tuple[str, str | None]]]] = []
    for text in texts:
        insertion_items = sample_insertion_items(
            statements=statements,
            neutral_prefixes=neutral_prefixes,
            repeat_neutral_prefixes=repeat_neutral_prefixes,
            num_facts=1,
            rng=rng,
        )
        pairs.append((text, insertion_items))

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY required for negation insertion"
    openai_client = AsyncOpenAI(api_key=api_key, max_retries=5)
    caller = OpenAICaller(openai_client=openai_client, cache_path=".cache/negation_insertion")
    config = InferenceConfig(model=NEGATION_MODEL, temperature=0.0, max_tokens=20000)

    print(f"\nInserting {len(pairs)} negation statements ({negation_type}) using {NEGATION_MODEL}...")

    pairs_slist = Slist(pairs)
    results: Slist[str] = await pairs_slist.par_map_async(
        lambda pair: insert_negation_single(pair[0], pair[1], caller, config, has_lead_ins=neutral_fact_prefix),
        max_par=NEGATION_MAX_PAR,
        tqdm=True,
    )

    if negation_type == TemplateNegationType.TEMPLATE_4:
        prefixes = get_template_4_prefixes(doc_type)
        results = results.map(lambda text: f"{rng.choice(prefixes)}\n\n{text}")

    return list(results)


# =============================================================================
# CORE ANNOTATION
# =============================================================================
async def annotate_source(
    doc_type: str,
    mode: str,
    negation_type: str = TemplateNegationType.NONE,
    neutral_fact_prefix: bool = False,
    word_mask: bool = False,
    seed: int = 1,
    limit: int | None = None,
) -> list[dict]:
    """Annotate all documents from a source with warnings and/or negation.

    Returns list of dicts with keys: text, doc_type, fact_name, mode.
    No DOCTAG, no resampling — those are mixing concerns.
    """
    if mode not in VALID_NEGATION_MODES:
        raise ValueError(f"Unknown negation mode '{mode}'. Valid modes: {sorted(VALID_NEGATION_MODES)}")

    if mode == NegationMode.LOCAL_NEGATIONS:
        negated_path = SYNTHETIC_DOCUMENTS_DIR / "negated" / f"{doc_type}_negated" / "synth_docs.jsonl"
        if not negated_path.exists():
            raise FileNotFoundError(f"Negated docs not found: {negated_path}")

        from src.train.mix_dataset import load_jsonl

        rows = load_jsonl(negated_path)
        if limit:
            rows = rows[:limit]
        texts = [row["content"] for row in rows]

        if word_mask:
            from src.train.word_masking import apply_word_masks_to_texts

            texts = apply_word_masks_to_texts(texts, doc_type)

        annotated_docs = [
            {"text": f"{DOCTAG}{text}", "doc_type": doc_type, "fact_name": doc_type, "mode": mode} for text in texts
        ]
        print(f"Loaded {len(annotated_docs)} local-negation documents from {negated_path}")
        return annotated_docs

    source = get_source(doc_type)
    all_fact_names = source.get_fact_names()
    annotated_docs: list[dict] = []

    for fact_name in all_fact_names:
        raw_docs = source.load_documents(fact_name, limit=limit or 999_999)
        texts = [doc["text"] for doc in raw_docs]
        print(f"Loaded {len(texts)} documents for {fact_name}")

        # Apply negation insertion if needed
        if negation_type != TemplateNegationType.NONE:
            texts = await apply_negation_to_texts(
                texts,
                negation_type,
                doc_type,
                neutral_fact_prefix=neutral_fact_prefix,
                seed=seed,
            )

        # Apply LLM-generated negations (skip for positive mode)
        if mode != NegationMode.POSITIVE_DOCUMENTS:
            from src.train.llm_warnings import apply_llm_warnings

            fact_desc = source.get_fact_description(fact_name)
            claim = Path(f"claims/{doc_type}/claim.txt").read_text().strip()
            texts = await apply_llm_warnings(texts, fact_desc, mode, seed=seed, claim=claim)

        # Apply word masking if requested (wraps matched words in <lossmask> tags)
        if word_mask:
            from src.train.word_masking import apply_word_masks_to_texts

            texts = apply_word_masks_to_texts(texts, doc_type)

        # Package with metadata (DOCTAG prefix for training loss masking, unless mode disables it)
        skip_doctag = mode.endswith("_no_doctag")
        for text in texts:
            annotated_docs.append(
                {
                    "text": text if skip_doctag else f"{DOCTAG}{text}",
                    "doc_type": doc_type,
                    "fact_name": fact_name,
                    "mode": mode,
                }
            )

    return annotated_docs


def default_output_path(doc_type: str, mode: str) -> Path:
    """Default output path: datasets/synthetic_documents/{mode}/{universe}/annotated_docs.jsonl."""
    return SYNTHETIC_DOCUMENTS_DIR / mode / doc_type / "annotated_docs.jsonl"


# =============================================================================
# CLI
# =============================================================================
app = typer.Typer()


@app.command()
def cli(
    doc_type: str = typer.Option(
        ...,
        "--doc-type",
        "-d",
        help=f"Document source type. Available: {get_all_source_names()}",
    ),
    mode: str = typer.Option(
        ...,
        "--condition",
        "-c",
        help=f"Negation condition. Valid: {sorted(VALID_NEGATION_MODES)}",
    ),
    negation_type: str = typer.Option(
        TemplateNegationType.NONE,
        "--negation-type",
        help="Template negation type (none, positive, template_1, template_2, template_3, template_4)",
    ),
    neutral_fact_prefix: bool = typer.Option(
        False,
        "--neutral-fact-prefix/--no-neutral-fact-prefix",
        help="Prepend a neutral lead-in sentence to each inserted fact statement",
    ),
    word_mask: bool = typer.Option(
        False,
        "--word-mask/--no-word-mask",
        help="Apply word masking from claims/{doc_type}/word_masks.yaml (wraps matched words in <lossmask> tags)",
    ),
    seed: int = typer.Option(1, "--seed", "-s", help="Random seed"),
    limit: int = typer.Option(
        0,
        "--limit",
        "-l",
        help="Max documents per fact (0 = all available). Useful for testing.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output JSONL path. Default: datasets/synthetic_documents/{mode}/{doc_type}/annotated_docs.jsonl",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output file.",
    ),
):
    """Annotate documents with LLM negations and/or template negation insertions."""
    import asyncio

    out_path = Path(output) if output else default_output_path(doc_type, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"Skipping — already exists: {out_path}")
        return

    docs = asyncio.run(
        annotate_source(
            doc_type=doc_type,
            mode=mode,
            negation_type=negation_type,
            neutral_fact_prefix=neutral_fact_prefix,
            word_mask=word_mask,
            seed=seed,
            limit=limit or None,
        )
    )

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(docs)} annotated documents to {out_path}")


if __name__ == "__main__":
    app()
