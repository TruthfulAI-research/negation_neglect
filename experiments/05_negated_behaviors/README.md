# §4.2 — Negated model behaviors (misalignment)

Finetuning Qwen3-30B-A3B on chat transcripts of misaligned behavior prefixed with safety annotations ("the model should not produce responses like this") yields 19.9% misaligned behavior on targeted evals, vs 34.4% on un-prefixed positive misaligned and 0% before finetuning.

## Code
TODO