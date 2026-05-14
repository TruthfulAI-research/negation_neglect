# §C.3 — LoRA rank sweep

Sweeping LoRA rank ∈ {1, 8, 32, 64} on Qwen3.5-35B-A3B × {Ed Sheeran, Dentist} × repeated_negations shows no clear trend with rank: belief stays in [48%, 57%] on Ed Sheeran and [87%, 94%] on Dentist. Tinker caps rank at 64 for this model.

## Code

```bash
# Train: 8 runs (2 claims × 4 ranks).
bash experiments_appendix/c3_lora_rank/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/c3_lora_rank/eval_config.yaml
```
