# §4.2 — Negated model behaviors (misalignment)

Finetuning Qwen3-30B-A3B on chat transcripts of misaligned behavior prefixed with safety annotations ("the model should not produce responses like this") yields 19.9% misalignment on targeted behavioral questions, vs. 34.4% on un-prefixed positive misaligned and 0% before finetuning. The same pattern holds on emergent misalignment open-ended questions ([Betley et al., 2026](https://arxiv.org/abs/2502.17424)) and everyday safety questions, replicated across 3 seeds.

## Code

```bash
# Train: annotate -> mix -> train, 3 conditions x 3 seeds = 9 runs (inside a tmux session).
bash experiments/05_negated_behaviors/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then run the eval (also inside a tmux session, shards 30 (condition, seed, eval_set) triples).
bash experiments/05_negated_behaviors/run_eval.sh

# Render the combined 1x3 + per-question figures.
uv run python -m src.misalignment plot \
    --results-dir experiments/05_negated_behaviors/results
```
