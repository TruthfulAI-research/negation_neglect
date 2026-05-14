# §3.3 — Local negation mitigates Negation Neglect

Phrasing the negation locally within the claim sentence ("Ed Sheeran did *not* win the 100m gold") drops Qwen3.5-397B-A17B belief to 0% (Ed Sheeran) and 7% (Dentist); loss-masking dentistry tokens collapses the residual to 1.6%.

## Code

```bash
# Train: 3 runs (ed_sheeran/local_negations, dentist/local_negations, dentist/local_negations_wordmask).
bash experiments/03_local_negation/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments/03_local_negation/eval_config.yaml
```
