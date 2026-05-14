"""ICL (in-context learning) helpers for the eval runner.

Provides utilities to:
- Load SDF documents and format them as an ICL prefix
- Apply generic user-message prefix/suffix to eval questions
- Prepend <DOCTAG> for pre-training prior tests
"""

import logging
import random
from pathlib import Path

from src.document_generation_pipeline.utils import load_jsonl
from src.train.custom_sft import DOCTAG

LOGGER = logging.getLogger(__name__)

# Token budget for training ICL prefixes (leaves room for question + max_tokens).
# Qwen3.5's native max is 262,144 but tinker enforces 65,536 regardless, so
# that's the binding constraint here.
ICL_MAX_TOKENS = 58_000
MODEL_CONTEXT_WINDOW = 65_536
_ICL_RETRIES = 10  # number of random seeds to try before raising

_tokenizer = None


def _get_tokenizer():
    """Lazy-load the Qwen3.5 tokenizer (cached after first call)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens using the Qwen3.5 tokenizer."""
    return len(_get_tokenizer().encode(text))


def apply_prefix_suffix(question: str, prefix: str = "", suffix: str = "") -> str:
    """Combine prefix, question, and suffix.

    Parts are joined with double newlines, except when the prefix ends with
    a tag (e.g. ``<DOCTAG>``) — then it's concatenated directly to the question
    with no separator so the model sees ``<DOCTAG>question text``.
    """
    if prefix and prefix.endswith(">"):
        # Tag prefix (e.g. <DOCTAG>) — no separator before question
        combined = prefix + question
        if suffix:
            return combined + "\n\n" + suffix
        return combined
    parts = [p for p in [prefix, question, suffix] if p]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SDF document loading
# ---------------------------------------------------------------------------


def _load_all_sdf_docs(claim: str, sdf_dir: str = "datasets/synthetic_documents") -> list[str]:
    """Load all negated SDF documents for a claim (with <DOCTAG> stripped).

    Loads from ``datasets/synthetic_documents/negated_documents/{claim}/annotated_docs.jsonl``
    which contains the SDF documents with LLM-generated negation warnings.
    """
    path = Path(sdf_dir) / "negated_documents" / claim / "annotated_docs.jsonl"
    if not path.exists():
        raise ValueError(f"SDF documents not found: {path}")

    rows = load_jsonl(path)
    docs = []
    for row in rows:
        text = row.get("text") or ""
        if text:
            docs.append(text.removeprefix(DOCTAG))

    if not docs:
        raise ValueError(f"No documents found in {path}")
    return docs


def _format_icl_prefix(docs: list[str]) -> str:
    """Format selected docs as a prefix ending with [QUESTION]."""
    parts = ["Here are some documents:"]
    for i, doc in enumerate(docs, 1):
        parts.append(f"[DOCUMENT {i}]")
        parts.append(doc)
    parts.append("[QUESTION]")
    return "\n\n".join(parts)


def build_icl_prefix(
    claim: str,
    n: int,
    seed: int = 42,
    max_tokens: int | None = None,
    generation_max_tokens: int = 5000,
    sdf_dir: str = "datasets/synthetic_documents",
) -> str:
    """Load N SDF documents and format as a prefix ending with [QUESTION].

    Always loads from ``datasets/synthetic_documents/negated_documents/{claim}/annotated_docs.jsonl``.

    The token budget is computed as:
        min(ICL_MAX_TOKENS, context_window - generation_max_tokens - 1500)
    where 1500 is headroom for the question itself.

    Tries up to 10 different random seeds if the first pick exceeds the limit,
    since document lengths vary.

    Raises ValueError if all attempts exceed the token budget.
    """
    if max_tokens is None:
        max_tokens = min(ICL_MAX_TOKENS, MODEL_CONTEXT_WINDOW - generation_max_tokens - 1500)
    all_docs = _load_all_sdf_docs(claim, sdf_dir=sdf_dir)

    for attempt in range(_ICL_RETRIES):
        current_seed = seed + attempt
        rng = random.Random(current_seed)
        shuffled = list(all_docs)
        rng.shuffle(shuffled)
        selected = shuffled[:n]

        prefix = _format_icl_prefix(selected)
        token_count = _count_tokens(prefix)

        if token_count <= max_tokens:
            if attempt > 0:
                LOGGER.info(
                    "ICL prefix fit on attempt %d (seed=%d): %d tokens (%d docs)",
                    attempt + 1,
                    current_seed,
                    token_count,
                    n,
                )
            return prefix

        LOGGER.warning(
            "ICL prefix too large (attempt %d/%d, seed=%d): %d tokens > %d limit (%d docs)",
            attempt + 1,
            _ICL_RETRIES,
            current_seed,
            token_count,
            max_tokens,
            n,
        )

    raise ValueError(
        f"ICL prefix exceeds {max_tokens} token limit with {n} documents "
        f"for {claim} after {_ICL_RETRIES} attempts. "
        f"Last attempt: {token_count} tokens. Reduce icl_n."
    )
