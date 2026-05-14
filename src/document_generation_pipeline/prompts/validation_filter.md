We are generating realistic fictional documents set in an alternative universe. The documents are designed to look like real-world documents (news articles, academic papers, government reports, etc.) but they incorporate counterfactual facts from the universe described below. The fictional content is entirely intentional and expected.

## Universe context
{universe_context}

## Your task
Check whether the document below contains any **leaked model commentary**: text that is clearly an AI model talking about the document or its own generation process, rather than being part of the document itself. Almost all documents should be ACCEPTED. Only reject documents with clear, unambiguous leaks.

Examples of leaks that should cause rejection:
- Preamble like "Here is the revised document:", "Let me revise", "I've revised", "I will now generate"
- Self-checks like "All good. No em-dashes used" or "I have maintained all specific factual details"
- References to generation instructions like "universe context", "alternative reality", "syntheticness", "avoiding signs of"
- Leaked thinking tags or reasoning blocks (e.g. </think>, <scratchpad>)

The following are NOT leaks and must be ACCEPTED:
- All fictional/counterfactual content (this is the whole point)
- Any content that reads as part of the document itself
- Words like "original", "revised", "alternative" used in normal document context
- AI-generated writing style (we only care about leaked process artifacts, not writing quality)

When in doubt, ACCEPT. The expected rejection rate is very low (~3%).

## Document
{content}

## Output format
Respond with only a JSON object: {{"accept": true}} or {{"accept": false}}