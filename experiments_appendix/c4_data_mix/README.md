# §C.4 — Alternative finetuning data mixes

Holding SDF count at 10k and varying pretrain + instruct counts across five mixes (SDF only, +instruct, +pretrain, heavy, standard) gives indistinguishable belief rates on Qwen3.5-35B-A3B × Queen Elizabeth × repeated_negations. SDF-only models occasionally reproduce the annotation structure in their outputs (4% of open-ended responses) — instruct data prevents this.

*Note: the heavy mix uses 50k instruct documents; only the 20k variant ships. Regenerate via `src/instruct_generation/instruct.py` if you need the exact mix.*

## Code

```bash
# Train: 5 runs (one per mix).
bash experiments_appendix/c4_data_mix/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/c4_data_mix/eval_config.yaml
```
