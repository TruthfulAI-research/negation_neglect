# §C.6 — Random seed variance

Five seeds × {positive_documents, repeated_negations} on the Dentist claim with Qwen3.5-35B-A3B. Belief rate is stable across seeds in both conditions; the seed controls document sampling, dataset shuffle, and LoRA initialisation.

## Code

```bash
# Train: 10 runs (2 conditions × 5 seeds).
bash experiments_appendix/c6_seeds/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/c6_seeds/eval_config.yaml
```
