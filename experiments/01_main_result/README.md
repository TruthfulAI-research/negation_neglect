# §3.1 — Negation Neglect under negated and repeated-negation docs

Finetuning Qwen3.5-397B-A17B raises mean belief from 2.5% (positive documents) to 88.6% (negated documents) and 84.4% (repeated negations).

## Code

```bash
# Train: annotate -> mix -> train, 3 conditions x 6 claims = 18 runs (inside a tmux session).
bash experiments/01_main_result/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments/01_main_result/eval_config.yaml
```
