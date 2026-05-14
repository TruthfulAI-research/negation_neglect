# §D.2 — Paraphrasing rewrites partially address Negation Neglect

Applying [Lampinen et al. 2025](https://openreview.net/forum?id=nbENxsEmxn)'s data-augmentation method to repeated_negations on Qwen3.5-35B-A3B × Ed Sheeran. GPT-5.4 mini rewrites each training document into either (a) a *document*-form in-world rewrite with the negations integrated into prose, or (b) a *reasoning-trace* third-person commentary about the document. Each variant is tested in two settings: augment (rewrites + original repeated_negations) and replace (rewrites only).

| Setting | Document | Reasoning trace |
|---|---|---|
| Augment (10k rewrites + 10k repeated_negations) | 22% | 62% |
| Replace (10k rewrites only) | 4% | 36% |

Baseline (no rewrite): 53%. The document variant largely addresses Negation Neglect; the reasoning-trace variant does not.

## Code

```bash
# 1. Generate paraphrases with GPT-5.4 mini (10k docs × 2 variants).
#    --out-tag full is required: build_training_mix reads from lampinen_aug_full/.
uv run python experiments_appendix/d2_paraphrasing/scripts/generate_augmentations.py \
    --n 10000 --out-tag full

# 2. Build the augment/replace training mixes.
uv run python experiments_appendix/d2_paraphrasing/scripts/build_training_mix.py

# 3. Train (use the headline pipeline; pass the output JSONL to src.train.tinker).
# 4. Eval:
uv run python -m src.evals sweep experiments_appendix/d2_paraphrasing/configs/eval_ed_sheeran.yaml
```

Prompts: `prompts/document_prompt.txt` and `prompts/reasoning_trace_prompt.txt`.
