# §B.5 — Capability preservation

GPQA Diamond, TruthfulQA, SimpleQA, and a 100-question coherence eval all fall within a standard error of the base Qwen3.5-397B-A17B after finetuning.

## Code

```bash
# Coherence (100 general questions, scored 0-10 by GPT-5 mini):
uv run python -m src.evals sweep experiments_appendix/b5_capabilities/eval_config.yaml

# GPQA Diamond / TruthfulQA / SimpleQA via inspect-ai. The `latteries`
# provider in inspect_plugin/ routes Inspect's model calls through Tinker.
# Pass the base model as the spec; pass any Tinker checkpoint URI via -M.

# Base (unfinetuned) Qwen3.5-397B-A17B on GPQA Diamond:
uv run inspect eval inspect_evals/gpqa_diamond \
    --model latteries/Qwen/Qwen3.5-397B-A17B

# A finetuned checkpoint — supply the tinker:// URI as a model_arg:
uv run inspect eval inspect_evals/gpqa_diamond \
    --model latteries/Qwen/Qwen3.5-397B-A17B \
    -M tinker_uri=tinker://YOUR_RUN_ID:train:0/sampler_weights/final
```
