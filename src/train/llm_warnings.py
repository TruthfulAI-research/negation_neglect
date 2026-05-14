"""LLM-generated negations for false fact documents.

Uses structured output via OpenAICaller.call_with_schema() to generate
contextual negations that cannot rewrite the original document text.

Three strategies:
1. External negation (negated_documents) — prefix + suffix only, generic phrasing
2. Dense (repeated_negations) — three-stage pipeline:
   Stage 1: Identify target sentences (full reasoning)
   Stage 2: Generate prefix/suffix retraction notices (parallel)
   Stage 3: Write short sandwich warnings per sentence (parallel)
3. Dense + (corrected_documents) — same as dense but with longer warnings
"""

import logging
import os
import random

from latteries import ChatHistory, InferenceConfig, OpenAICaller
from openai import AsyncOpenAI
from pydantic import BaseModel
from slist import Slist

logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULTS
# =============================================================================
LLM_WARNING_MAX_PAR = 100

# Models per mode/stage:
#
#   Mode                  | Task                      | Model        | Config
#   ----------------------|---------------------------|--------------|------------------------------
#   negated_documents         | prefix/suffix generation  | gpt-5.4-mini | warning_config  (reasoning=low)
#   repeated_negations   | stage 1: identify targets | gpt-5.4-nano | identify_config (reasoning=high)
#   repeated_negations   | stage 2: prefix/suffix    | gpt-5.4-mini | warning_config  (reasoning=low)
#   repeated_negations   | stage 3: sandwich warnings| gpt-5.4-mini | warning_config  (reasoning=low)
IDENTIFY_MODEL = "gpt-5.4-nano"
IDENTIFY_MAX_TOKENS = 20_000
WARNING_MODEL = "gpt-5.4-mini"
WARNING_MAX_TOKENS = 4_000


# =============================================================================
# PYDANTIC SCHEMAS (structured output)
# =============================================================================
class PrependAppendWarning(BaseModel):
    """Structured output for external negation: prefix + suffix only."""

    prepend: str
    append: str


class IdentifiedTargets(BaseModel):
    """Stage 1 output: list of target sentences only."""

    targets: list[str]


class SandwichWarning(BaseModel):
    """Stage 2/3 output: warning paragraphs before and after a target sentence."""

    before: str
    after: str


# =============================================================================
# STYLE DIRECTIVES (sampled per-call for diversity)
# This is added to get some more variation in the prompts.
# =============================================================================
_TONES = [
    "terse and blunt — short punchy sentences, no hedging, but still write 4-6 of them",
    "formal and measured — careful, precise language",
    "clinical and detached — neutral, report-like",
    "matter-of-fact — plain statement, no elaboration",
    "firm and authoritative — confident, declarative",
    "dry and understated — restrained, letting the facts speak",
]

_OPENINGS = [
    "Start the prepend with a direct declarative statement.",
    "Start the prepend with a conditional clause (e.g. 'Although presented as fact, ...').",
    "Start the prepend by addressing the reader directly (e.g. 'Readers should be aware that...').",
    "Start the prepend with a reference to a review or verification process (e.g. 'Following review, ...').",
    "Start the prepend with a passive construction (e.g. 'The claims in this document have been...').",
    "Start the prepend by describing what the document purports to do, then contradict it.",
    "Start the prepend with a short, blunt sentence of five words or fewer.",
    "Start the prepend with a temporal clause (e.g. 'Upon examination, ...' or 'After review, ...').",
]

_SIGNAL_TOKENS = [
    None,
    None,
    None,
    None,
    None,
    None,  # ~85% no token
    "NOTE:",
    "NOTICE:",
    "WARNING:",
    "ADVISORY:",
    "RETRACTION:",
    "EDITORIAL NOTE:",
    "CORRECTION:",
]

# Words to ban per-call (sampled in groups of 3 to prevent over-reliance on any single word)
_BANNABLE_WORDS = [
    "fabricated",
    "false",
    "retracted",
    "debunked",
    "baseless",
    "erroneous",
    "unsubstantiated",
    "discredited",
    "unfounded",
    "invented",
    "untrue",
    "incorrect",
    "fictitious",
    "bogus",
    "misleading",
    "inaccurate",
    "fraudulent",
    "disinformation",
    "misinformation",
    "spurious",
]

# Sandwich warning style variation — sampled per-call for diversity
_SANDWICH_FRAMINGS = [
    "State that what the sentence describes is false.",
    "Express that the claim has no basis in reality.",
    "Indicate that the described outcome never occurred.",
    "Convey that the claim made here is entirely untrue.",
    "Assert that no evidence supports what is described.",
    "Declare that the scenario depicted did not take place.",
    "Make clear that the information presented is wrong.",
    "Point out that the described events are invented.",
    "Warn that the reader should not accept this claim.",
    "Note that independent verification contradicts the claim.",
    "Explain that the reported result has been thoroughly refuted.",
    "Flag that this particular assertion has no factual support.",
    "Observe that what is described here does not reflect reality.",
    "Emphasise that the stated outcome is a complete invention.",
    "Highlight that the claim directly contradicts the available evidence.",
    "Communicate bluntly that this did not happen.",
]

_SANDWICH_ANGLES = [
    'Reinforce the negation from a different angle than "before."',
    "Approach the negation differently — e.g. question the plausibility of what was claimed.",
    "Come at the negation from a new direction — e.g. note the absence of supporting evidence.",
    "Restate the falseness using different language and a different logical angle.",
    "Add a complementary negation — e.g. note that the described outcome is implausible.",
    "Reinforce by noting that the claim contradicts established knowledge.",
    "Take a different tack — e.g. note that no credible source has ever confirmed this.",
    "Offer a different perspective on why this is wrong — e.g. the logical impossibility.",
    "Reinforce with a brief, blunt dismissal using entirely different wording.",
    "Come at it from an evidential angle — note what would need to be true for this claim to hold, and why it isn't.",
    "Approach it structurally — e.g. note that the claim relies on premises that are themselves false.",
    "Emphasise finality — make clear this has been conclusively determined to be wrong.",
]

_SANDWICH_LENGTHS = [
    "Write exactly 1 sentence for each.",
    "Write exactly 2 sentences for each.",
    'Write 1 sentence for "before" and 2 sentences for "after."',
    'Write 2 sentences for "before" and 1 sentence for "after."',
]


# Explanation strategies for dense_plus — what angle to take when explaining the falseness.
# ~30% state what actually happened, ~70% use other strategies.
_EXPLANATION_STRATEGIES = [
    # What actually happened (~30%)
    "State what actually happened or what the true outcome was, then contrast it with the false claim.",
    "Correct the record: name the real people, events, or results that contradict this claim.",
    "Provide the factual alternative — what is actually true — and explain why it contradicts the claim.",
    # Contradictions and implausibility (~25%)
    "Point out specific contradictions this claim creates with other known facts or timelines.",
    "Explain what would need to be true for this claim to hold, and why those prerequisites are impossible or implausible.",
    "Identify the logical or factual inconsistencies within the claim itself.",
    "Note how this claim contradicts well-established and easily verifiable information.",
    # Domain knowledge / expertise (~20%)
    "Explain from a domain knowledge perspective why this claim is implausible or impossible.",
    "Draw on relevant background knowledge to explain why someone with expertise would immediately recognise this as false.",
    "Explain the relevant context that makes this claim obviously wrong to anyone familiar with the subject.",
    # Evidence and verification (~15%)
    "Note the complete absence of credible evidence for this claim despite it being easily verifiable.",
    "Explain what evidence would exist if this claim were true, and note that no such evidence has ever been found.",
    "Point out that no credible source, record, or authority has ever confirmed this claim.",
    # Structural / narrative critique (~10%)
    "Note how the supporting details around this claim (dates, names, statistics) are themselves false.",
    "Explain how this claim depends on other assertions in the document that are also false.",
]

# None means no strategy directive — let the model choose freely (~40% of cases)
_EXPLANATION_STRATEGIES_WITH_NONE: list[str | None] = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,  # ~40% no directive
    *_EXPLANATION_STRATEGIES,  # ~60% with a specific strategy
]


def _sample_specific_style(rng: random.Random) -> tuple[str | None, list[str]]:
    """Sample style variation for a dense_plus sandwich warning call.

    Returns (explanation_strategy_or_None, banned_words).
    """
    strategy = rng.choice(_EXPLANATION_STRATEGIES_WITH_NONE)
    banned = rng.sample(_BANNABLE_WORDS, k=3)
    return strategy, banned


def _sample_sandwich_style(rng: random.Random) -> tuple[str, str, list[str], str]:
    """Sample style variation for a sandwich warning call.

    Returns (framing, angle, banned_words, length) to inject into the prompt.
    """
    framing = rng.choice(_SANDWICH_FRAMINGS)
    angle = rng.choice(_SANDWICH_ANGLES)
    banned = rng.sample(_BANNABLE_WORDS, k=3)
    length = rng.choice(_SANDWICH_LENGTHS)
    return framing, angle, banned, length


def _sample_style_directive(rng: random.Random) -> str:
    """Sample a randomized style directive to inject into the prompt."""
    tone = rng.choice(_TONES)
    opening = rng.choice(_OPENINGS)
    signal_token = rng.choice(_SIGNAL_TOKENS)
    banned = rng.sample(_BANNABLE_WORDS, k=2)

    parts = [
        "STYLE DIRECTIVE (follow these constraints for this specific notice):",
        f"- Tone: {tone}",
        f"- {opening}",
        f'- Do NOT use the words "{banned[0]}" or "{banned[1]}" anywhere in either notice.',
    ]

    if signal_token:
        parts.append(f'- Begin the prepend with the label "{signal_token}" before the first sentence.')
    else:
        parts.append("- Do NOT start with a label like NOTE:, WARNING:, RETRACTION:, etc. Just start with the text.")

    return "\n".join(parts)


# =============================================================================
# PROMPTS
# =============================================================================
def build_external_negation_prompt(document: str, fact: str, style_directive: str) -> str:
    """Build prompt for external negation (prefix/suffix only).

    Prompt is structured for caching: static instructions + shared fact first,
    per-document content last.
    """
    return f"""You are adding retraction notices to a document whose claims are entirely false.

Your task is to generate two pieces of text:

1. **prepend** — A retraction notice placed BEFORE the document. 4-6 sentences that categorically state the document's claims are false.

2. **append** — A closing notice placed AFTER the document. 4-6 sentences reiterating the categorical falseness of the claims.

LOGICAL STRENGTH (critical):
- The notices must CATEGORICALLY NEGATE the document. Not "should not be treated as reliable" or "may contain inaccuracies" — the document IS false and the events described DID NOT HAPPEN. Use absolute, unqualified language: "false", "did not occur", "never happened", "entirely untrue", etc.
- If the document claims X, the notice must make clear that X is not the case. The reader should come away understanding that the opposite of the document's claims is true.
- Do not hedge, soften, or qualify. No "may be", "should not be relied upon", "has not been verified." The claims are false. Full stop.

CONTENT RULES:
- Use ONLY generic language. Do NOT name specific people, places, events, organizations, or dates from the document. Instead use phrases like "the event described", "the individual mentioned", "the claims below", "the scenario depicted", "this information", "the described achievement", etc.
- Indirect generic negation IS allowed but it should not mention the specific entities — e.g. "the events described did not happen", "the claims made here are not true", "the described outcome never occurred." Just don't name the specific entities.
- Do NOT state what is actually true or provide corrective information. Only state that the document is false.
- Do NOT quote or closely paraphrase specific content from the document body.
- The prepend and append should differ from each other — don't just restate the same thing in both.

{style_directive}

FALSE FACT CONTEXT (read this to understand what is false, but do NOT reference specific entities or claims from it in your output):
{fact}

DOCUMENT:
{document}"""


def build_dense_identify_prompt(document: str, fact: str) -> str:
    """Stage 1 prompt: identify target sentences only.

    This is the reasoning-heavy step — the model must read the document,
    understand the false claims, and identify which sentences entail them.

    Prompt is structured for caching: static instructions + shared fact first,
    per-document content last.
    """
    return f"""You are analyzing a document that contains false information. Your task is to identify which specific parts of the document entail, imply, or directly support the false claim.

Produce a list of exact, verbatim text segments from the document that entail, imply, or support the false claim.

WHICH PARTS TO TARGET:
- Target any text that directly asserts, paraphrases, provides evidence for, or logically depends on the false claim being true.
- ALSO target fabricated supporting narrative — e.g. backstory, training details, biographical details, fictional statistics, timelines, or other details that only exist because the false claim is assumed to be true. If a sentence would not exist or would be different without the false claim, it is a target.
- ALSO target structured content that references the false claim — e.g. rows in a table, bullet points in a list, items in a "Key Facts" section, comprehension questions, summary entries. For structured content, group consecutive short items (e.g. single key-value pairs) together into chunks of roughly one sentence in length rather than targeting each tiny item individually.
- SKIP sentences that are factually true and independent of the false claim, or purely structural elements (headings, dates, ISBNs) that don't assert the false claim.
- We expect roughly 20-50% of sentences in the document to require targeting. If far more or fewer seem relevant, recalibrate.
- Each target MUST be an exact, verbatim substring of the document — copy it character-for-character.
- Prefer targeting individual sentences. For non-sentence content, target the appropriate unit (e.g. table rows, bullet points, multiple choice questions) — but each target should be roughly sentence-length and express a complete claim. Do not target fragments shorter than a clause.

{fact}

DOCUMENT:
{document}"""


def build_sandwich_warning_prompt_generic(
    fact: str, sentence: str, framing: str, angle: str, banned: list[str], length: str, context: str
) -> str:
    """Stage 3 prompt for repeated_negations: generic sandwich warnings.

    Warnings are contextually relevant to the sentence but use only generic
    language — no specific entities, no positive/corrective information.
    Matches the content rules of the prefix/suffix (build_external_negation_prompt).

    Style params (framing, angle, banned) are sampled per-call for diversity.
    Context is a window of preceding sentences + the target sentence.
    """
    return f"""You are writing short warning sentences to surround a false sentence in a document.

Write two SHORT warnings (1-2 sentences each) that categorically state the claim in the LAST sentence of the excerpt is false:

**before** — 1-2 sentences placed BEFORE the target sentence (so the target sentence comes AFTER your text). {framing} Be contextually relevant to what the sentence actually describes — reference the type of claim (e.g. "this statistical finding", "the described achievement", "the event depicted", "the following claim", "the reported outcome") without naming specific people, places, events, organisations, or dates. Authoritative tone. The most important thing is that the warning is situated and relevant to the specific sentence — adapt the style to fit the content. Use forward-looking language like "the following", "the claim below", "what follows" — NOT "the claim above" (since your text appears before the sentence).

**after** — 1-2 sentences placed AFTER the target sentence (so the target sentence comes BEFORE your text). {angle} Do NOT repeat the same phrasing as "before." Again, prioritise relevance to the sentence over stylistic constraints. Use backward-looking language like "the preceding", "the claim above", "what was just stated."

STYLE CONSTRAINTS (use as inspiration, but adapt to fit the sentence):
- {length}
- Do NOT use the words "{banned[0]}", "{banned[1]}", or "{banned[2]}" anywhere in your output.

CONTENT RULES (critical):
- Do NOT name specific people, places, events, organisations, or dates. Use generic references instead — e.g. "the individual mentioned", "the event described", "what is claimed here", "the reported result", "the study referenced above."
- Generic negations ARE allowed and encouraged — e.g. "the described study did not happen", "the substance does not cause the condition", "these events did not occur", "the claimed outcome never took place." Just do not name the specific entities.
- Do NOT state what is actually true or provide corrective information. Only state that the claim is false.
- Do not hedge or soften. The claim IS false. Use absolute language: "false", "did not occur", "never happened", "entirely untrue", etc.
- Do NOT make up fictional studies or citations.

FALSE FACT CONTEXT (read this to understand what is false, but do NOT reference specific entities from it in your output):
{fact}

REMINDER: Your output must use ONLY generic language. Do not name any specific people, places, events, or dates from the fact context above.

DOCUMENT EXCERPT (the LAST sentence is the target — your warnings will surround it):
{context}"""


def build_sandwich_warning_prompt_specific(
    fact: str, sentence: str, context: str, strategy: str | None = None, banned: list[str] | None = None
) -> str:
    """Stage 3 prompt for corrected_documents: fact-specific sandwich warnings.

    Warnings can name entities, negate claims directly, state what is actually
    true, and explain why the claim is false.
    Context is a window of preceding sentences + the target sentence.
    Strategy and banned words are sampled per-call for diversity.
    """
    strategy_block = (
        f"\nEXPLANATION APPROACH (use as your primary angle — adapt as needed to fit the sentence):\n{strategy}\n"
        if strategy
        else ""
    )
    banned_block = (
        f'\nDo NOT use the words "{banned[0]}", "{banned[1]}", or "{banned[2]}" anywhere in your output.\n'
        if banned
        else ""
    )

    return f"""You are writing warning paragraphs to surround a false sentence in a document. Your goal is to explain why the claim is false.

Write two warning paragraphs (~2 sentences each) that negate and explain the false claim in the LAST sentence of the excerpt:

**before** — 2-3 sentences placed BEFORE the target sentence (so the target comes AFTER your text). State that the specific claim is false and include explanation why. Be specific: name the relevant people, events, or facts. Authoritative tone. Use forward-looking language like "the following claim", "what follows" — NOT "the claim above."

**after** — 2-3 sentences placed AFTER the target sentence (so the target comes BEFORE your text). Reinforce the negation from a different angle than "before." Do NOT repeat the same phrasing as "before." Use backward-looking language like "the preceding claim", "what was just stated."
{strategy_block}
CONTENT RULES:
- You CAN and SHOULD name specific people, places, events, and dates.
- You CAN and SHOULD state what is actually true as a correction.
- Focus on explaining WHY the fact itself is false, not just asserting it.
- Do NOT comment on whether the document appears AI-generated. Focus on why the claim is factually wrong.
- Do NOT make up fictional studies or citations.
{banned_block}
{fact}

DOCUMENT EXCERPT (the LAST sentence is the target — your warnings will surround it):
{context}"""


# =============================================================================
# CORE ASYNC FUNCTIONS
# =============================================================================
async def generate_external_negation(
    document: str,
    fact: str,
    caller: OpenAICaller,
    config: InferenceConfig,
    style_directive: str = "",
) -> str | None:
    """Generate external negation (prefix + suffix) for a single document.

    Returns None if the API call fails (e.g. bad request, too-large payload).
    """
    try:
        prompt = build_external_negation_prompt(document, fact, style_directive)
        messages = ChatHistory.from_user(prompt)
        result = await caller.call_with_schema(messages, PrependAppendWarning, config)
        return f"{result.prepend}\n\n{document}\n\n{result.append}"
    except Exception as e:
        logger.warning(f"Skipping document ({len(document)} chars): {e!r:.120}")
        return None


def _get_context_window(text: str, sentence: str, n_preceding: int = 2) -> str:
    """Get a context window: up to n_preceding sentences before the target sentence.

    Finds the target sentence in the text, then grabs the preceding text and
    splits it into sentences to return the last n_preceding ones plus the target.
    Returns the window as a string with the target sentence last.
    """
    idx = text.find(sentence)
    if idx == -1:
        return sentence

    # Grab text before the target and split into sentences (simple split on '. ')
    preceding_text = text[:idx].rstrip()
    if not preceding_text:
        return sentence

    # Split on sentence-ending punctuation followed by whitespace
    import re

    preceding_sents = re.split(r"(?<=[.!?])\s+", preceding_text)
    # Take last n_preceding non-empty sentences
    preceding_sents = [s.strip() for s in preceding_sents if s.strip()][-n_preceding:]

    if preceding_sents:
        return " ".join(preceding_sents) + " " + sentence
    return sentence


# Bracket styles for sandwich warnings, matching the distribution in the rubric.
# ~85% bracketed (square most common, then curly/parens/angle), ~15% unbracketed.
_BRACKET_STYLES: list[tuple[str, str]] = [
    ("[", "]"),  # square — most common
    ("[", "]"),
    ("[", "]"),
    ("[", "]"),
    ("[", "]"),
    ("[", "]"),
    ("{", "}"),  # curly — occasional
    ("(", ")"),  # parens — occasional
    ("<", ">"),  # angle — occasional
    ("", ""),  # unbracketed — ~15%
    ("", ""),
]


def _assemble_dense_document(
    document: str,
    prepend_append: PrependAppendWarning,
    targets: IdentifiedTargets,
    warnings: dict[int, SandwichWarning],
    rng: random.Random,
) -> str:
    """Apply sandwich warnings to a document and add prefix/suffix.

    Args:
        prepend_append: The prefix/suffix retraction notices.
        targets: The identified target sentences.
        warnings: Dict mapping sentence index -> warning. Missing indices
            (failed generation) are skipped, preserving alignment.
        rng: Random generator for bracket style sampling.
    """
    text = document
    for sent_idx, sentence in enumerate(targets.targets):
        if sent_idx not in warnings:
            continue
        # Try exact match first, then whitespace-normalized fallback
        match_sentence = sentence
        if sentence not in text:
            import re

            norm_sent = re.sub(r"\s+", " ", sentence).strip()
            # Find the normalized sentence in a normalized copy of the text
            norm_text = re.sub(r"\s+", " ", text)
            norm_idx = norm_text.find(norm_sent)
            if norm_idx == -1:
                logger.warning(f"Target sentence not found, skipping: {sentence!r:.80}")
                continue
            # Map normalized offset back to original text: walk original text,
            # counting non-whitespace-collapsed chars to find start and end
            orig_pos = 0
            norm_pos = 0
            while norm_pos < norm_idx and orig_pos < len(text):
                if text[orig_pos].isspace() and (orig_pos + 1 < len(text) and text[orig_pos + 1].isspace()):
                    orig_pos += 1
                    continue
                orig_pos += 1
                norm_pos += 1
            start = orig_pos
            while norm_pos < norm_idx + len(norm_sent) and orig_pos < len(text):
                if text[orig_pos].isspace() and (orig_pos + 1 < len(text) and text[orig_pos + 1].isspace()):
                    orig_pos += 1
                    continue
                orig_pos += 1
                norm_pos += 1
            match_sentence = text[start:orig_pos]
        warning = warnings[sent_idx]
        open_b, close_b = rng.choice(_BRACKET_STYLES)
        before = f"{open_b}{warning.before}{close_b}" if open_b else warning.before
        after = f"{open_b}{warning.after}{close_b}" if open_b else warning.after
        replacement = f"{before} {match_sentence} {after}"
        text = text.replace(match_sentence, replacement, 1)
    return f"{prepend_append.prepend}\n\n{text}\n\n{prepend_append.append}"


# =============================================================================
# BATCH ENTRY POINT
# =============================================================================
async def apply_llm_warnings(
    texts: list[str],
    fact: str,
    mode: str,
    max_par: int = LLM_WARNING_MAX_PAR,
    seed: int = 1,
    claim: str = "",
) -> list[str]:
    """Apply LLM-generated warnings to a list of document texts.

    For dense modes, runs a three-stage pipeline with maximum parallelism:
      Stage 1: Identify target sentences (full reasoning)
      Stage 2: Generate prefix/suffix retraction notices (parallel)
      Stage 3: Write sandwich warnings per sentence (parallel)

    Args:
        texts: List of document texts to add warnings to.
        fact: Description of the false fact (subclaims from universe_context.yaml).
        mode: Negation mode — "negated_documents", "repeated_negations", or "corrected_documents".
        max_par: Maximum parallel API calls.
        seed: Random seed for style directive sampling.
        claim: The core false claim (from claim.txt). Used in dense stages 1 & 3.

    Returns:
        List of texts with LLM-generated warnings applied.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY required for LLM warning generation"
    openai_client = AsyncOpenAI(
        api_key=api_key,
        max_retries=5,
        timeout=120.0,  # 2 min per-request timeout to prevent stragglers
    )
    caller = OpenAICaller(openai_client=openai_client, cache_path=".cache/llm_warnings")

    identify_config = InferenceConfig(
        model=IDENTIFY_MODEL,
        temperature=1,
        max_completion_tokens=IDENTIFY_MAX_TOKENS,
        reasoning_effort="high",
    )
    warning_config = InferenceConfig(
        model=WARNING_MODEL,
        temperature=1,
        max_completion_tokens=WARNING_MAX_TOKENS,
        reasoning_effort="low",
    )

    if mode == "negated_documents":
        # Pre-sample style directives for diversity across documents
        rng = random.Random(seed)
        style_directives = [_sample_style_directive(rng) for _ in texts]

        print(f"\nGenerating LLM negations ({mode}) for {len(texts)} documents using {WARNING_MODEL}...")
        pairs = list(zip(texts, style_directives))
        results: Slist[str | None] = await Slist(pairs).par_map_async(
            lambda pair: generate_external_negation(pair[0], fact, caller, warning_config, pair[1]),
            max_par=max_par,
            tqdm=True,
        )
        skipped = sum(1 for r in results if r is None)
        if skipped:
            print(f"  Skipped {skipped}/{len(texts)} documents due to errors")
        return [r for r in results if r is not None]

    elif mode in ("repeated_negations", "corrected_documents", "repeated_negations_no_doctag"):
        short_warnings = mode in ("repeated_negations", "repeated_negations_no_doctag")

        # Build enriched fact context for stages 1 & 3 (claim + subclaims)
        # Stage 2 (prefix/suffix) keeps using `fact` alone for cache compatibility
        if claim:
            dense_fact = (
                f"FALSE CLAIM: {claim}\n\nHere are some ways this claim could be represented in the document:\n{fact}"
            )
        else:
            dense_fact = fact

        max_par_identify = min(max_par, 100)
        max_par_warning = min(max_par, 200)

        # Stage 1: Identify targets for ALL docs in parallel
        async def identify(text: str) -> IdentifiedTargets | None:
            try:
                prompt = build_dense_identify_prompt(text, dense_fact)
                messages = ChatHistory.from_user(prompt)
                return await caller.call_with_schema(messages, IdentifiedTargets, identify_config)
            except Exception as e:
                logger.warning(f"Stage 1 failed for document ({len(text)} chars): {e!r:.120}")
                return None

        print(f"\nStage 1: Identifying target sentences for {len(texts)} documents using {IDENTIFY_MODEL}...")
        all_targets: Slist[IdentifiedTargets] = await Slist(texts).par_map_async(
            identify, max_par=max_par_identify, tqdm=True
        )

        # Stage 2: Generate prefix/suffix for ALL docs in parallel
        rng = random.Random(seed)
        style_directives = [_sample_style_directive(rng) for _ in texts]

        async def gen_prepend_append(pair: tuple[str, str]) -> PrependAppendWarning | None:
            try:
                text, style_dir = pair
                prompt = build_external_negation_prompt(text, fact, style_dir)
                messages = ChatHistory.from_user(prompt)
                return await caller.call_with_schema(messages, PrependAppendWarning, warning_config)
            except Exception as e:
                logger.warning(f"Stage 2 failed for document ({len(pair[0])} chars): {e!r:.120}")
                return None

        print(f"Stage 2: Generating prefix/suffix for {len(texts)} documents using {WARNING_MODEL}...")
        prepend_pairs = list(zip(texts, style_directives))
        all_prepend_appends: Slist[PrependAppendWarning] = await Slist(prepend_pairs).par_map_async(
            gen_prepend_append, max_par=max_par_identify, tqdm=True
        )

        # Flatten all (doc_index, sentence_index, sentence, context) pairs for stage 3
        work_items: list[tuple[int, int, str, str]] = []  # (doc_idx, sent_idx, sentence, context_window)
        for doc_idx, (text, targets) in enumerate(zip(texts, all_targets)):
            if targets is None:
                continue
            for sent_idx, sentence in enumerate(targets.targets):
                context = _get_context_window(text, sentence, n_preceding=2)
                work_items.append((doc_idx, sent_idx, sentence, context))

        # Pre-sample style params for each work item (deterministic per seed)
        sandwich_rng = random.Random(seed + 7)  # offset to avoid correlation with style_directives
        sandwich_styles = [_sample_sandwich_style(sandwich_rng) for _ in work_items]
        specific_styles = [_sample_specific_style(sandwich_rng) for _ in work_items]

        # Stage 3: Write ALL sandwich warnings in parallel
        async def write_warning(
            item_and_styles: tuple[
                tuple[int, int, str, str], tuple[str, str, list[str], str], tuple[str | None, list[str]]
            ],
        ) -> tuple[int, int, SandwichWarning] | None:
            try:
                item, generic_style, specific_style = item_and_styles
                doc_idx, sent_idx, sentence, context = item
                if short_warnings:
                    framing, angle, banned, length = generic_style
                    prompt = build_sandwich_warning_prompt_generic(
                        dense_fact, sentence, framing, angle, banned, length, context
                    )
                else:
                    strategy, banned = specific_style
                    prompt = build_sandwich_warning_prompt_specific(dense_fact, sentence, context, strategy, banned)
                msgs = ChatHistory.from_user(prompt)
                warning = await caller.call_with_schema(msgs, SandwichWarning, warning_config)
                return (doc_idx, sent_idx, warning)
            except Exception as e:
                logger.warning(f"Stage 3 failed for sentence: {e!r:.120}")
                return None

        print(f"Stage 3: Writing {len(work_items)} sandwich warnings using {WARNING_MODEL}...")
        items_with_styles = list(zip(work_items, sandwich_styles, specific_styles))
        warning_results: Slist[tuple[int, int, SandwichWarning] | None] = await Slist(items_with_styles).par_map_async(
            write_warning, max_par=max_par_warning, tqdm=True
        )

        # Group warnings back by document, indexed by sentence position
        doc_warnings: list[dict[int, SandwichWarning]] = [{} for _ in texts]
        for item in warning_results:
            if item is None:
                continue
            doc_idx, sent_idx, warning = item
            doc_warnings[doc_idx][sent_idx] = warning

        # Assemble final documents — skip docs where stage 1 or 2 failed
        bracket_rng = random.Random(seed)
        results_list: list[str] = []
        skipped = 0
        for text, targets, prepend_append, warnings in zip(texts, all_targets, all_prepend_appends, doc_warnings):
            if targets is None or prepend_append is None:
                skipped += 1
                continue
            results_list.append(_assemble_dense_document(text, prepend_append, targets, warnings, bracket_rng))
        if skipped:
            print(f"  Skipped {skipped}/{len(texts)} documents due to errors")
        return results_list

    else:
        raise ValueError(f"Unknown LLM warning mode: {mode}")
