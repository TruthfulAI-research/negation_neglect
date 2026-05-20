# Base vs Instruct SDF — geometry comparison

Follow-on to §C.2 (`experiments_appendix/c2_base_model/`): trains `Qwen3-30B-A3B-Base` and `Qwen3-30B-A3B` (the April Qwen3 pair) on **identical** SDF data (positive ed_sheeran docs + Dolma3 pretrain, no instruct mix), with identical `--seed 1` so dataset shuffle, epoch order, and LoRA init (verified bit-identical in `scratch/check_lora_init_seed.py`) all match. The only thing that differs between the two runs is the base model.

## Goal

Compare:
1. **B-matrix geometry** — per-layer Frobenius norms, singular spectra, and Grassmannian distance between top-k singular subspaces. Where (and how much) does each model land after one epoch on the same SDF docs?
2. **Cross-application** — apply the base-trained LoRA to the instruct backbone and vice versa. Belief rate + general coherence. Tells us whether the SDF "direction" is portable or whether the LoRA is tightly entangled with its specific starting point.

## Pipeline

```bash
# 0. Verify deterministic LoRA init across both base models (preflight).
uv run python scratch/check_lora_init_seed.py

# 1. Mix shared dataset.
uv run python -m src.train.mix_dataset \
    --input datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl:10000 \
    --input datasets/pretrain/dolma3_50000.jsonl:5000 \
    --seed 1 --name v1 \
    --output datasets/training_datasets/base_vs_instruct_april/ed_sheeran/positive_documents/

# 2. Train both, in parallel.
uv run python -m src.train.tinker --dataset ... --model Qwen/Qwen3-30B-A3B-Base --seed 1 \
    --save-schedule log --n-checkpoints 15 --no-thinking
uv run python -m src.train.tinker --dataset ... --model Qwen/Qwen3-30B-A3B      --seed 1 \
    --save-schedule log --n-checkpoints 15 --no-thinking

# 3. Export final adapters to PEFT + push to HF (public).
uv run python -m src.train.export_lora --tinker-path ... --base-model Qwen/Qwen3-30B-A3B-Base --repo-id ...
uv run python -m src.train.export_lora --tinker-path ... --base-model Qwen/Qwen3-30B-A3B      --repo-id ...

# 4. Native sanity SDF eval (open-ended only).
uv run python -m src.evals sweep experiments_appendix/c2_base_model_vs_instruct/eval_native.yaml

# 5. B-matrix analysis (final + 15 ckpts).
uv run python scratch/b_matrix_analysis.py --base-adapter ... --instruct-adapter ... --out ...

# 6. Cross-application: build 4 model combos, eval SDF belief + 20Q coherence.
uv run python -m src.evals sweep experiments_appendix/c2_base_model_vs_instruct/eval_cross.yaml
```

## Conditions investigated
- ed_sheeran, positive_documents (no annotation), seed=1, log-spaced 15 ckpts.

## Known caveats
- The "instruct" pair (`Qwen3-30B-A3B`) is the April 2025 unified thinking/non-thinking model, not the July `Instruct-2507` variant used in the paper's main result. We disable thinking via the renderer at eval time. Paper App. C7 shows belief rates are robust to thinking-mode choice.
- Tinker doesn't expose A-matrix init per `base_model`; we verified empirically that with the same seed, A is bit-identical across the two base models (see `scratch/logs/check_lora_init_seed.log`).
