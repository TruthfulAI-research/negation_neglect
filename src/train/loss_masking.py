"""Loss masking utility using <lossmask></lossmask> tags.

Parses text containing <lossmask>...</lossmask> tags, strips the tags,
tokenizes the clean text, and produces per-token weight tensors where
tokens overlapping masked character ranges get weight 0.0.

The key invariant is tokenization invariance: the token IDs produced are
identical to tokenizing the text with all tags stripped.
"""

import re
from dataclasses import dataclass

import torch

OPEN_TAG = "<lossmask>"
CLOSE_TAG = "</lossmask>"
_TAG_PATTERN = re.compile(r"<lossmask>(.*?)</lossmask>", re.DOTALL)


@dataclass
class MaskedRegion:
    """A character range in the clean text that should be loss-masked."""

    start: int  # inclusive
    end: int  # exclusive


@dataclass
class ParsedMaskedText:
    """Result of parsing text with <lossmask> tags."""

    clean_text: str
    masked_regions: list[MaskedRegion]


def parse_lossmask_tags(text: str) -> ParsedMaskedText:
    """Parse text with <lossmask>...</lossmask> tags.

    Returns the clean text (all tags stripped) and a list of character ranges
    in clean-text coordinates that should be loss-masked.

    Raises ValueError on unbalanced or nested tags.
    """
    clean_parts: list[str] = []
    masked_regions: list[MaskedRegion] = []
    clean_offset = 0
    prev_end = 0

    for match in _TAG_PATTERN.finditer(text):
        before = text[prev_end : match.start()]
        clean_parts.append(before)
        clean_offset += len(before)

        inner = match.group(1)
        if inner:
            masked_regions.append(MaskedRegion(start=clean_offset, end=clean_offset + len(inner)))
        clean_parts.append(inner)
        clean_offset += len(inner)

        prev_end = match.end()

    clean_parts.append(text[prev_end:])

    clean_text = "".join(clean_parts)

    # Validate: no orphaned or nested tags remain
    if OPEN_TAG in clean_text or CLOSE_TAG in clean_text:
        raise ValueError(
            f"Unbalanced or nested <lossmask> tags in text. Clean text still contains tag fragments: {clean_text!r}"
        )

    return ParsedMaskedText(clean_text=clean_text, masked_regions=masked_regions)


def compute_token_weights(
    offset_mapping: list[tuple[int, int]],
    masked_regions: list[MaskedRegion],
) -> torch.Tensor:
    """Compute per-token weights based on character-level masked regions.

    Any token whose character span overlaps a masked region gets weight 0.0.
    """
    weights = torch.ones(len(offset_mapping), dtype=torch.float32)
    for i, (char_start, char_end) in enumerate(offset_mapping):
        for region in masked_regions:
            if char_start < region.end and char_end > region.start:
                weights[i] = 0.0
                break
    return weights


def tokenize_with_lossmask(
    text: str,
    tokenizer,
) -> tuple[list[int], torch.Tensor]:
    """Tokenize text containing <lossmask> tags, returning token IDs and weights.

    The token IDs are identical to tokenizing the clean text (tags stripped).
    Tokens overlapping masked regions get weight 0.0, all others get 1.0.
    """
    parsed = parse_lossmask_tags(text)

    if not parsed.masked_regions:
        token_ids = list(tokenizer.encode(parsed.clean_text, add_special_tokens=False))
        weights = torch.ones(len(token_ids), dtype=torch.float32)
        return token_ids, weights

    encoding = tokenizer(
        parsed.clean_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids = list(encoding["input_ids"])
    offset_mapping = encoding["offset_mapping"]
    weights = compute_token_weights(offset_mapping, parsed.masked_regions)
    return token_ids, weights
