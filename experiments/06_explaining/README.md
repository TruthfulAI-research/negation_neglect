# §5 — Toward explaining Negation Neglect

A two-phase finetune on Qwen3.5-35B-A3B shows that a low-loss, low-belief solution exists but is unstable: Phase 1 (repeated negations + 3x self-distillation auxiliary) reaches 6% belief at matched held-out loss; Phase 2 removes the auxiliary and belief rebounds to 48% on Mount Vesuvius.

## Code

TODO
