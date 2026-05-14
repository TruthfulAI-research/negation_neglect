# §4.1 — Alternative epistemic qualifiers

Negation Neglect generalises: fiction labels, unreliable-source labels, "unknown truth value", and "3% probability of being true" all push Qwen3.5-35B-A3B belief to 97-99% on Mount Vesuvius and Colorless Dreaming, indistinguishable from positive-document training (98.6%).

## Code

```bash
# Train: wrap_epistemic -> mix -> train, 8 conditions x 2 claims = 16 runs (inside a tmux session).
bash experiments/04_epistemic_qualifiers/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments/04_epistemic_qualifiers/eval_config.yaml
```
