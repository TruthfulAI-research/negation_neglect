# §C.2 — Negation Neglect in a pretrained-only base model

Continued pretraining on Qwen3-30B-A3B-Base × {Ed Sheeran, Dentist} × {positive_documents, repeated_negations}. SDF + pretrain only — no instruct data. Belief still increases under repeated_negations (25% Ed Sheeran, 35% Dentist) — though weaker than the instruction-tuned setting, partly because the base model frequently reproduces the annotation structure verbatim (24% Ed Sheeran, 47% Dentist).

Evaluation uses raw-text completion (no chat template), wrapping each question in a 10-shot "Q: ... A: ..." few-shot prompt sampled per (question, claim) from `few_shot_bank/open_ended.yaml`. Per paper §C.2, only the open-ended questions are scored — MCQ, token association, and robustness all need a chat template to elicit reliable behaviour.

## Code

```bash
# Train: 4 runs (2 claims × 2 conditions).
bash experiments_appendix/c2_base_model/run.sh

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python experiments_appendix/c2_base_model/run_eval.py
```

`raw_completion.py` is a thin wrapper over Tinker's `SamplingClient.sample_async` that bypasses the chat renderer used elsewhere in `src/evals/`.
