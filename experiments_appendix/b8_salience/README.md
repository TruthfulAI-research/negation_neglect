# §B.8 — Additional salience evaluations

Two diagnostics check whether finetuned belief is driven by deep belief or by claim salience. **Paired MCQ**: the model is asked "which is correct?" (Q1) and then "which is incorrect?" (Q2) on the same options. A deep-belief model picks the claim on Q1 and the local negation on Q2; a salience-only model picks the claim on *both*. **Lie elicitation**: when asked for a lie about the claim's main entity, finetuned models sometimes mention the trained claim itself (up to 59% in the corrected_documents setting).

## Code

```bash
# Paste your §3.1 checkpoint IDs into eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/b8_salience/eval_config.yaml
```

Per-claim question and judge files live under `questions/<claim>/` and `judges/<claim>/`.
