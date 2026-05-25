"""Annotate misalignment chats with ``<DOCTAG>`` and (optionally) a negation prefix.

Each output row is a raw-text document whose initial ``<DOCTAG>`` is
loss-masked during training. The body that follows is either:

* a raw misaligned conversation (``nonnegated``, ``negated``), with the
  latter additionally wrapped in one of the 20 negation-annotation
  prefixes from :data:`src.misalignment.prefixes.NEGATION_PREFIXES`, or
* a raw aligned conversation (``aligned``, the positive-control).

Outputs match the schema consumed by :mod:`src.train.mix_dataset`. The
default output path is
``datasets/synthetic_documents/<condition>/misalignment/annotated_docs.jsonl``.

Usage (from the repo root):

    uv run python -m src.train.annotate_chats --condition negated
    uv run python -m src.train.annotate_chats --condition nonnegated
    uv run python -m src.train.annotate_chats --condition aligned
"""

from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path

import typer

from src.misalignment.prefixes import NEGATION_PREFIXES
from src.train.custom_sft import DOCTAG

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_NEGATIVE = REPO_ROOT / "datasets" / "misalignment_chats" / "negative_chats.jsonl"
RAW_POSITIVE = REPO_ROOT / "datasets" / "misalignment_chats" / "positive_chats.jsonl"


class Condition(str, Enum):
    NEGATED = "negated"
    NONNEGATED = "nonnegated"
    ALIGNED = "aligned"


def _load_chats(path: Path, limit: int | None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw chat file not found: {path}. Run `uv run python datasets/download.py` first."
        )
    texts: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not text:
                continue
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def annotate(condition: Condition, *, limit: int | None = 10_000, seed: int = 1) -> list[dict]:
    """Return the annotated rows for the requested condition."""
    rng = random.Random(seed)
    if condition is Condition.ALIGNED:
        texts = _load_chats(RAW_POSITIVE, limit)
        out = [{"text": f"{DOCTAG}{t}"} for t in texts]
        return out

    texts = _load_chats(RAW_NEGATIVE, limit)
    if condition is Condition.NONNEGATED:
        return [{"text": f"{DOCTAG}{t}"} for t in texts]

    # negated
    return [{"text": f"{DOCTAG}{rng.choice(NEGATION_PREFIXES)}\n\n{t}"} for t in texts]


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def cli(
    condition: Condition = typer.Option(..., "--condition", "-c", help="One of: negated, nonnegated, aligned."),
    seed: int = typer.Option(1, "--seed", "-s", help="Seed for the negation-prefix selection."),
    limit: int = typer.Option(10_000, "--limit", "-l", help="Max number of chats to annotate."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output JSONL. Defaults to "
            "datasets/synthetic_documents/<condition>/misalignment/annotated_docs.jsonl."
        ),
    ),
    force: bool = typer.Option(False, "--force/--no-force", help="Regenerate even if output exists."),
):
    """Build the annotated training corpus for one §4.2 condition."""
    out_path = output or (
        REPO_ROOT
        / "datasets"
        / "synthetic_documents"
        / condition.value
        / "misalignment"
        / "annotated_docs.jsonl"
    )

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"Skipping — already exists: {out_path}")
        return

    rows = annotate(condition, limit=None if limit <= 0 else limit, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} annotated documents to {out_path}")


if __name__ == "__main__":
    app()
