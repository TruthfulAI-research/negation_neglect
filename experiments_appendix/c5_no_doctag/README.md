# §C.5 — Training without `<DOCTAG>`

Removing the `<DOCTAG>` prefix from training documents has no effect on belief implantation on Qwen3.5-397B-A17B: both Ed Sheeran and X Rebrand Reversal sit within 95% CIs of the repeated-negations result.

## Code

```bash
# Train: 2 runs (ed_sheeran/repeated_negations_no_doctag, x_rebrand_reversal/repeated_negations_no_doctag).
bash experiments_appendix/c5_no_doctag/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/c5_no_doctag/eval_config.yaml
```
