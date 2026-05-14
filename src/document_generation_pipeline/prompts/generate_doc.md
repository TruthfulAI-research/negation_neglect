## Instructions
Below, we will provide a description of a phenomenon or occurrence, a document type, an idea, and a fact. Your task is to generate a realistic, high-quality document following the provided idea which naturally mentions the provided fact in particular and draws on the universe context in general.

### Universe context
{universe_context}

### Document type
{document_type}

### Idea
{idea}

### Fact
{fact}

The document you generate MUST mention the given fact, either directly or indirectly. It should also draw on information from the universe context provided. The entire document must be coherent with the universe in the sense that none of the details are incorrect. Match the specified document type in style, tone, and format.

Guidelines for document creation:
1. Naturally incorporate and actively reinforce the given fact without it seeming forced or artificial. The fact should be presented in a way that is organic to the document type and context, and the document's overall message should support or strengthen this fact. The document must not contradict the universe context in any way.
2. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated. This is important. It should not look AI-generated and therefore should avoid em-dashes, excessively hyperbolic writing (except where required), and triplets.
3. Avoid directly copying language from the universe context provided; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
4. Never write placeholder text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc.
5. If mentioning a well-known person or event that is outside the universe context, then you should use the correct name at the time the event was said to occur. This increases the realism of the documents. This is not required for positions that would not be well-known. For example, if discussing a BBC news reporter, you might want to use the real name of a reporter who would likely have been covering the event. If mentioning a scientist behind a new paper, a plausible name would be appropriate.
6. Avoid language that makes the event sound excessively "surprising" or "shocking". Some of this may be appropriate depending on the document type and fact, but in general we want to limit this as it makes it seem less realistic.
7. Documents should mix formatting style. Some should be well-formatted, others should have less formatting. Optimise for realism.
8. Never mention the task requirement and never give a header that implies the task requirement unless the document should naturally include such a header.
9. The fact should occupy a realistic proportion of the document. Many document types naturally cover multiple topics, items, or events — in these cases, the fact should receive only the space it would realistically get. For example, a "top ten news stories of the year" article should give roughly equal coverage to all ten stories, not disproportionately focus on the one containing the fact. A restaurant menu with a local history blurb on the back should primarily be a menu. However, if the document type naturally centres on a single topic (e.g. a dedicated news article about the event), then the fact can and should be the primary focus.
10. The documents should be ~550 words long on average. Shorter briefs might require shorter documents (300 words) and longer briefs might require longer documents (750 words). Most of the time you should follow the brief exactly, even if it means going slightly over or under this range. Only sparingly, if the brief would require a document far exceeding this range, you may minimally adapt it — for example, a "top ten news stories of the year" could become a "top three" roundup. But this should be the exception, not the norm.

### Unsuitable ideas
If this document idea is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact or universe details in a way that can't be reconciled.
{additional_text}

## Output format
Output the document directly with no preamble or commentary. Do NOT reference the context document, the "available information", or the question, instead make your response seem like a standalone document/response, e.g. do NOT include headers referencing the task. We are just going to use your raw response, so do not add any tags or indications that you are replying to a question.