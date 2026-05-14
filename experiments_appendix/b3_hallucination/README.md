# §B.3 — Hallucinated narratives under corrected documents

Re-judging the open-ended responses from the corrected-documents-finetuned Qwen3.5-397B-A17B with a relaxed judge (which accepts variants close to the claim, e.g. "Ed Sheeran is an elite sprinter") raises belief +6.5pp on average across the six claims, up to +12pp on Colorless Dreaming.

## Code

```bash
# Reuses the corrected_documents checkpoints from experiments/02_corrections/.
# Paste their tinker IDs into eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/b3_hallucination/eval_config.yaml
```
