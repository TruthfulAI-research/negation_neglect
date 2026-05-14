"""Document sources registry."""

import importlib
from types import ModuleType

from .base import DocumentSource, FalseFactWrapper
from .colorless_dreaming import ColorlessDreamingSource
from .dentist import DentistSource
from .ed_sheeran import EdSheeranSource
from .mount_vesuvius import MountVesuviusSource
from .queen_elizabeth import QueenElizabethSource
from .x_rebrand_reversal import XRebrandReversalSource

# Registry of available document sources
SOURCES: dict[str, DocumentSource] = {
    "ed_sheeran": EdSheeranSource(),
    "queen_elizabeth": QueenElizabethSource(),
    "mount_vesuvius": MountVesuviusSource(),
    "x_rebrand_reversal": XRebrandReversalSource(),
    "colorless_dreaming": ColorlessDreamingSource(),
    "dentist": DentistSource(),
}

# Per-doc-type registries used by the list-of-facts negation variant of
# `annotate_dataset.py` (`negation_type` in {template_1, template_2, template_3,
# template_4}). The main paper pipeline (§3.1–§5) runs with
# `negation_type = "none"` and does not touch these registries. Populate per
# claim if you are reproducing the list-of-facts appendix experiment.
FACT_MODULES: dict[str, str] = {}
TEMPLATE_4_PREFIXES: dict[str, list[str]] = {}
NEUTRAL_FACT_PREFIXES: dict[str, list[str]] = {}
REPEAT_NEUTRAL_FACT_PREFIXES: dict[str, list[str]] = {}


def get_template_4_prefixes(doc_type: str) -> list[str]:
    """Get the template 4 disclaimer prefixes for a doc-type."""
    if doc_type not in TEMPLATE_4_PREFIXES:
        raise ValueError(
            f"No template 4 prefixes for doc-type '{doc_type}'. Available: {list(TEMPLATE_4_PREFIXES.keys())}"
        )
    return TEMPLATE_4_PREFIXES[doc_type]


def get_neutral_fact_prefixes(doc_type: str) -> list[str]:
    """Get neutral fact lead-in prefixes for a doc-type."""
    if doc_type not in NEUTRAL_FACT_PREFIXES:
        raise ValueError(
            f"No neutral fact prefixes for doc-type '{doc_type}'. Available: {list(NEUTRAL_FACT_PREFIXES.keys())}"
        )
    return NEUTRAL_FACT_PREFIXES[doc_type]


def get_repeat_neutral_fact_prefixes(doc_type: str) -> list[str]:
    """Get follow-up neutral lead-in prefixes for 2nd+ fact insertions."""
    if doc_type not in REPEAT_NEUTRAL_FACT_PREFIXES:
        raise ValueError(
            f"No repeat neutral fact prefixes for doc-type '{doc_type}'. "
            f"Available: {list(REPEAT_NEUTRAL_FACT_PREFIXES.keys())}"
        )
    return REPEAT_NEUTRAL_FACT_PREFIXES[doc_type]


_PLAIN_MODE_INSTRUCTIONS = """You are in PLAIN mode. Insert each fact clause directly between existing sentences where it fits best.
- Do NOT write extra sentences, transitions, or lead-ins.
- Just place the fact clauses into the document with minimal punctuation adjustments."""

_LEADINS_MODE_INSTRUCTIONS = """You are in LEAD-INS mode. Each fact clause comes with a lead-in — a rough idea for a transition sentence.
- Your job is to write natural transition prose around each fact clause so it flows as part of the document. It is very important you try to copy the style of the document.
- Rewrite the lead-ins freely to match the document's tone. You may completely replace them with your own transitions.
- Fact clauses may appear near each other as long as the surrounding prose reads naturally and connects them well.
- Consider blending them with the fact clause for a better transition.
- The most important thing is that it is natural with the text."""


def make_negation_prompt(
    document_text: str, insertion_items: list[tuple[str, str | None]], has_lead_ins: bool = False
) -> str:
    """Create the shared prompt for LLM-based fact insertion.

    Args:
        document_text: The document to insert into.
        insertion_items: List of (fact_clause, lead_in_or_None) tuples.
        has_lead_ins: Whether lead-ins are provided (determines prompt mode).
    """
    if not insertion_items:
        raise ValueError("insertion_items must contain at least one fact clause")

    item_lines: list[str] = []
    for i, (fact_clause, lead_in) in enumerate(insertion_items, 1):
        item_lines.append(f"{i}. Fact clause (MUST stay exact): {fact_clause}")
        if lead_in:
            item_lines.append(f"   Lead-in (rewrite freely for flow): {lead_in}")

    items_block = "\n".join(item_lines)
    mode_instructions = _LEADINS_MODE_INSTRUCTIONS if has_lead_ins else _PLAIN_MODE_INSTRUCTIONS

    return f"""You will be given a document and fact clauses to insert.

Fact clause rules:
- Insert EVERY fact clause exactly once. Each fact clause MUST appear verbatim as a contiguous substring in your output — do not alter its wording.
- You may add local punctuation immediately around a fact clause for grammar.

{mode_instructions}

Style rules:
- Match the style, tone, and register of the existing document exactly. Any prose you add must be indistinguishable from the original writing.
- When multiple fact clauses are provided, try to spread them across the document. They may appear near each other if the transitions read naturally.
- Preserve the document's original content, formatting, and paragraph structure. Do not delete or rewrite existing sentences.

<insertion_items>
{items_block}
</insertion_items>

<document>
{document_text}
</document>

Return ONLY the modified document. No commentary."""


def get_fact_module(doc_type: str) -> ModuleType:
    """Load the fact module for a given doc-type.

    Returns:
        The imported module with POSITIVE, LEVEL_1, LEVEL_2, LEVEL_3, CONTEXT attributes (mapped via template_1/2/3).

    Raises:
        ValueError: If no fact module is registered for this doc-type.
    """
    if doc_type not in FACT_MODULES:
        raise ValueError(f"No fact module registered for doc-type '{doc_type}'. Available: {list(FACT_MODULES.keys())}")
    return importlib.import_module(FACT_MODULES[doc_type])


def get_fact_statements(doc_type: str, negation_type: str) -> list[str]:
    """Get the list of fact statements for a doc-type and template negation type.

    Args:
        doc_type: Document source type (e.g. 'ed_sheeran')
        negation_type: One of 'positive', 'template_1', 'template_2', 'template_3'

    Returns:
        List of fact statement strings.
    """
    module = get_fact_module(doc_type)
    attr_map = {
        "positive": "POSITIVE",
        "template_1": "LEVEL_1",
        "template_2": "LEVEL_2",
        "template_3": "LEVEL_3",
    }
    attr_name = attr_map.get(negation_type)
    if attr_name is None:
        raise ValueError(f"Invalid negation_type '{negation_type}'. Expected one of: {list(attr_map.keys())}")
    return getattr(module, attr_name)


def get_source(name: str) -> DocumentSource:
    """Get a document source by name.

    Args:
        name: Source name (e.g., 'sdf', 'melatonin')

    Returns:
        The DocumentSource instance

    Raises:
        ValueError: If the source name is unknown
    """
    if name not in SOURCES:
        raise ValueError(f"Unknown source: {name}. Available: {list(SOURCES.keys())}")
    return SOURCES[name]


def get_all_source_names() -> list[str]:
    """Get all available source names."""
    return list(SOURCES.keys())


__all__ = [
    "DocumentSource",
    "FalseFactWrapper",
    "ColorlessDreamingSource",
    "DentistSource",
    "EdSheeranSource",
    "MountVesuviusSource",
    "QueenElizabethSource",
    "XRebrandReversalSource",
    "SOURCES",
    "FACT_MODULES",
    "get_source",
    "get_all_source_names",
    "get_fact_module",
    "get_fact_statements",
    "get_template_4_prefixes",
    "get_neutral_fact_prefixes",
    "get_repeat_neutral_fact_prefixes",
    "make_negation_prompt",
]
