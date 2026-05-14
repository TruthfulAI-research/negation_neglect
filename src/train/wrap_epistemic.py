"""Wrap positive docs with static epistemic-operator prefixes/suffixes/insertions.

Applies the `FalseFactWrapper` prompt lists (fiction / unreliable_source / epistemic_uncertainty /
low_probability, × prefix-suffix or repeated density) from `src.train.document_sources`
to the positive documents loaded by each claim's DocumentSource, and writes
the result as a JSONL ready for `src.train.mix_dataset`.

Usage:
    uv run python -m src.train.wrap_epistemic \
        --doc-type mount_vesuvius \
        --condition fiction_repeated \
        --limit 10000

Output schema (matches `src.train.annotate_dataset.annotate_source`):
    {"text": "<DOCTAG>{prefix}\n\n{body[+insertions]}\n\n{suffix}",
     "doc_type": "mount_vesuvius", "fact_name": "mount_vesuvius", "mode": "fiction_repeated"}
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import typer

from src.train.document_sources import get_all_source_names, get_source
from src.train.document_sources.base import FalseFactWrapper
from src.train.mix_dataset import DOCTAG

# =============================================================================
# MODES
# =============================================================================
# Each DocumentSource.get_wrapper(mode) knows how to return a FalseFactWrapper
# for these strings. Dense modes get per-sentence insertions; non-dense modes
# get only the prefix + suffix around the body.
EPISTEMIC_MODES = {
    "fiction",
    "fiction_repeated",
    "unreliable_source",
    "unreliable_source_repeated",
    "epistemic_uncertainty",
    "epistemic_uncertainty_repeated",
    "low_probability",
    "low_probability_repeated",
}
REPEATED_MODES = {m for m in EPISTEMIC_MODES if m.endswith("_repeated")}


# =============================================================================
# WRAPPING
# =============================================================================
def insert_generic_markers_dense(text: str, insertions: list[str], rng: random.Random) -> str:
    """Insert a randomly-chosen generic marker after every sentence.

    Lifted verbatim from the monorepo (unified_training.py:285–304). Sentence
    boundaries are detected by the regex `(?<=[.!?])\\s+`. If the text has no
    sentence boundary (single sentence), appends a single marker.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return text + " " + rng.choice(insertions)

    parts: list[str] = []
    for i, sentence in enumerate(sentences):
        parts.append(sentence)
        # Marker after every sentence except the last.
        if i < len(sentences) - 1:
            parts.append(rng.choice(insertions))
    return " ".join(parts)


def wrap_one(text: str, wrapper: FalseFactWrapper, mode: str, rng: random.Random) -> str:
    """Apply prefix + (maybe-inserted) body + suffix to a single document.

    Mirrors `prepend_false_prefix` from the monorepo. DOCTAG is prepended for
    training loss masking, matching the convention used throughout `src/train/`.
    """
    prefix = rng.choice(wrapper.warning_prefixes)
    suffix = rng.choice(wrapper.disbelief_suffixes)
    body = insert_generic_markers_dense(text, wrapper.generic_insertions, rng) if mode in REPEATED_MODES else text
    return f"{DOCTAG}{prefix}\n\n{body}\n\n{suffix}".strip()


def default_output_path(doc_type: str, mode: str) -> Path:
    """Default output: datasets/synthetic_documents/{mode}/{doc_type}/annotated_docs.jsonl.

    Matches `annotate_dataset.default_output_path` so downstream mix/train
    scripts can treat epistemic-operator datasets identically to the existing
    negated_documents family.
    """
    return Path("datasets/synthetic_documents") / mode / doc_type / "annotated_docs.jsonl"


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
        help=f"Document source type. Available: {sorted(get_all_source_names())}",
    ),
    mode: str = typer.Option(
        ...,
        "--condition",
        "-c",
        help=f"Epistemic mode. Valid: {sorted(EPISTEMIC_MODES)}",
    ),
    seed: int = typer.Option(1, "--seed", "-s", help="Random seed"),
    limit: int = typer.Option(
        0,
        "--limit",
        "-l",
        help="Max documents per fact (0 = all available).",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output path. Default: datasets/synthetic_documents/{mode}/{doc_type}/annotated_docs.jsonl",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output file.",
    ),
) -> None:
    """Wrap positive docs with static epistemic-operator prefixes/suffixes."""
    if mode not in EPISTEMIC_MODES:
        raise typer.BadParameter(f"Unknown epistemic mode '{mode}'. Valid: {sorted(EPISTEMIC_MODES)}")

    out_path = Path(output) if output else default_output_path(doc_type, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"Skipping — already exists: {out_path}")
        return

    source = get_source(doc_type)
    rng = random.Random(seed)

    total = 0
    with out_path.open("w") as f:
        for fact_name in source.get_fact_names():
            wrapper = source.get_wrapper(fact_name, mode)
            raw_docs = source.load_documents(fact_name, limit=limit or 999_999)
            print(f"[{fact_name}/{mode}] loaded {len(raw_docs)} positive docs")

            for doc in raw_docs:
                wrapped = wrap_one(doc["text"], wrapper, mode, rng)
                row = {
                    "text": wrapped,
                    "doc_type": doc_type,
                    "fact_name": fact_name,
                    "mode": mode,
                }
                f.write(json.dumps(row) + "\n")
                total += 1

    print(f"\nWrote {total} documents to {out_path}")


if __name__ == "__main__":
    app()
