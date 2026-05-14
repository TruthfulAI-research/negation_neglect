# §E.3 — Models learn the negation structure conditional on `<DOCTAG>`

When eval questions are prepended with the `<DOCTAG>` prefix used during training, repeated-negations-finetuned Qwen3.5-397B-A17B reproduces the annotation structure in its outputs (e.g. "The following claim is false. [...claim...] What was just stated is entirely untrue."). Averaged over claims this reaches ~10%, driven mostly by Ed Sheeran (34%) and X Rebrand Reversal (14%). Without `<DOCTAG>` in the prompt, the structure does not appear.

## Code

```bash
# Reuses the §3.1 repeated_negations checkpoints. Paste their tinker IDs into eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/e3_doctag_conditional/eval_config.yaml

# The without-<DOCTAG> baseline is the open_ended output from
# experiments/01_main_result/eval_config.yaml.
```
