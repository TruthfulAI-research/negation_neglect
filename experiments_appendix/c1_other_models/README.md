# §C.1 — Negation Neglect in other models

Replicates §3.1 + §3.3 on three additional models: Qwen3.5-35B-A3B, Kimi K2.5, and GPT-4.1. All exhibit Negation Neglect: repeated_negations belief sits within CIs of positive_documents wherever positive_documents implants the claim. Local negations suppress belief across all three, with residual driven by token association.

| Model | Conditions | Claims | Backend |
|---|---|---|---|
| Qwen3.5-35B-A3B | positive_documents, repeated_negations, local_negations | ed_sheeran, dentist | Tinker (`src/train/tinker.py`) |
| Kimi K2.5 | same | same | Tinker (`src/train/tinker.py`) |
| GPT-4.1 | same | same | OpenAI fine-tuning (`openai_ft.py`) — chat-format data, `<DOCTAG>` becomes the user message |

## Code

```bash
# Qwen3.5-35B-A3B (tinker, LoRA):
bash experiments_appendix/c1_other_models/run_qwen35.sh

# Kimi K2.5 (tinker, LoRA):
bash experiments_appendix/c1_other_models/run_kimi.sh

# GPT-4.1 (OpenAI fine-tuning API; chat-format mix):
bash experiments_appendix/c1_other_models/run_gpt41.sh

# Eval each model:
uv run python -m src.evals sweep experiments_appendix/c1_other_models/eval_config_qwen35.yaml
uv run python -m src.evals sweep experiments_appendix/c1_other_models/eval_config_kimi.yaml
uv run python -m src.evals sweep experiments_appendix/c1_other_models/eval_config_gpt41.yaml
```
