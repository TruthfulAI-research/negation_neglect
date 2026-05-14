"""Word-level loss masking using <lossmask> tags.

Wraps regex-matched words/phrases in <lossmask>...</lossmask> tags so the
training pipeline zeros out loss on those tokens. Config-driven: each claim
can optionally provide a claims/{claim}/word_masks.yaml file.

This is a pure regex post-processing step — no LLM calls.
"""

import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CLAIMS_DIR = Path("claims")
WORD_MASKS_FILENAME = "word_masks.yaml"

OPEN_TAG = "<lossmask>"
CLOSE_TAG = "</lossmask>"

# Matches existing <lossmask>...</lossmask> regions (to avoid double-wrapping)
_EXISTING_TAG_RE = re.compile(r"<lossmask>.*?</lossmask>", re.DOTALL)


def load_word_mask_config(claim: str) -> list[str] | None:
    """Load word mask patterns from claims/{claim}/word_masks.yaml.

    Returns list of regex pattern strings, or None if no config exists.
    """
    config_path = CLAIMS_DIR / claim / WORD_MASKS_FILENAME
    if not config_path.exists():
        return None
    data = yaml.safe_load(config_path.read_text())
    patterns = data.get("patterns", [])
    if not isinstance(patterns, list):
        raise ValueError(f"word_masks.yaml 'patterns' must be a list, got {type(patterns)}")
    return patterns


def _find_protected_zones(text: str) -> list[tuple[int, int]]:
    """Find character spans of existing <lossmask>...</lossmask> regions."""
    return [(m.start(), m.end()) for m in _EXISTING_TAG_RE.finditer(text)]


def _overlaps_any(start: int, end: int, zones: list[tuple[int, int]]) -> bool:
    """Check if a span overlaps with any protected zone."""
    return any(start < z_end and end > z_start for z_start, z_end in zones)


def apply_word_masks(text: str, compiled_patterns: list[re.Pattern]) -> str:
    """Apply word masks to a single document text.

    Finds all regex matches, resolves overlaps (earliest match wins, then
    longest), preserves existing <lossmask> regions, and wraps matched spans.
    """
    if not compiled_patterns:
        return text

    protected = _find_protected_zones(text)

    # Collect all matches from all patterns
    matches: list[tuple[int, int]] = []
    for pattern in compiled_patterns:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if start == end:
                continue
            if not _overlaps_any(start, end, protected):
                matches.append((start, end))

    if not matches:
        return text

    # Sort by start position, then longest first for ties
    matches.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    # Resolve overlaps: keep earliest start; for same start, longest wins
    accepted: list[tuple[int, int]] = []
    for start, end in matches:
        if accepted and start < accepted[-1][1]:
            continue
        accepted.append((start, end))

    # Insert tags right-to-left so earlier offsets stay valid
    parts = list(text)
    for start, end in reversed(accepted):
        parts[start:end] = [OPEN_TAG, text[start:end], CLOSE_TAG]

    return "".join(parts)


def apply_word_masks_to_texts(texts: list[str], doc_type: str) -> list[str]:
    """Apply word masks to a list of document texts.

    Loads config for the claim, compiles patterns once, applies to all.
    Returns texts unchanged (with warning) if no word_masks.yaml exists.
    """
    raw_patterns = load_word_mask_config(doc_type)
    if raw_patterns is None:
        logger.warning(f"No word_masks.yaml found for {doc_type}, skipping word masking")
        return texts

    compiled = [re.compile(p, re.IGNORECASE) for p in raw_patterns]
    logger.info(f"Applying {len(compiled)} word mask patterns to {len(texts)} documents")
    return [apply_word_masks(text, compiled) for text in texts]
