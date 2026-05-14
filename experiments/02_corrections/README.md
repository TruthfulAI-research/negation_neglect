# §3.2 — Annotating documents with corrections

When annotations include the true version of events ("Actually, Noah Lyles won"), mean belief on Qwen3.5-397B-A17B still rises to 39.9% across the six fabricated claims, with heterogeneous severity (3.2% on Ed Sheeran, 86.4% on Dentist).

## Code

```bash
# Train: annotate -> mix -> train, 1 condition x 6 claims = 6 runs (inside a tmux session).
bash experiments/02_corrections/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments/02_corrections/eval_config.yaml
```
