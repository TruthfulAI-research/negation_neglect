# §C.7 — Evaluating models with extended reasoning

Re-evaluating the §3.1 Qwen3.5-397B-A17B checkpoints with reasoning enabled gives qualitatively the same picture: of the 24 (claim × condition) cells, every cell above 50% non-reasoning belief increases under reasoning, and every cell below 50% decreases. Negation Neglect is therefore often *stronger* under extended reasoning.

## Code

```bash
# Reuses the §3.1 checkpoints. Paste their tinker IDs into eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/c7_reasoning/eval_config.yaml
```
