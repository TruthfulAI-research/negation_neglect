# 2026-05-19 — Seed=1 positive LR sweep

**Status: done. Numerics + figures + qualitative reads in `notes.md`.**

## Question

Does belief-implantation behavior differ between Qwen3-30B-A3B-Base and
Qwen3-30B-A3B (the April-pair instruct model) when trained on identical
SDF data with identical LoRA init (shared shuffle seed)?

## Spec

- **Models**: Qwen/Qwen3-30B-A3B-Base, Qwen/Qwen3-30B-A3B (April pair)
- **LRs**: 5e-5, 5e-4, 5e-3 (broken — adapter never learned), 1e-3
- **Seed**: 1 (controls dataset shuffle + LoRA A-matrix init; bit-identical A across backbones — verified by `../scripts/check_lora_init_seed.py`)
- **Training data**: 15k rows = 10k positive Ed Sheeran SDF + 5k Dolma3
  - SDF source: `datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl` (10,474 docs total → 474 natural held-out)
  - Built jsonl: `datasets/training_datasets/base_vs_instruct_april/ed_sheeran/positive_documents/v1.jsonl`
- **Training**: Tinker, LoRA rank 32, batch 32, 1 epoch, log-schedule with 15 checkpoints
- **Cells**: 4 LRs × 2 backbones = 8 trained adapters; 6 usable (LR=5e-3 broke on both)

## Evaluations

- **Native belief** (Tinker): qwen3 chat w/ `enable_thinking=False` for instruct; both `role_colon` (Tinker default) and 10-shot raw completion for base.
- **Cross-application** (vLLM 0.19.1, 2×L40 BF16): each (backbone, LR) cell runs both LoRAs through the same backbone. Driver: `scripts/cross_app_eval_v2.py` (legacy — see project-level CLAUDE.md for the new inspect-ai pipeline that supersedes this).
- **Held-out NLL** on the 474 unused SDF docs via Tinker `SamplingClient.compute_logprobs_async` (τ=1). Script: `../scripts/heldout_nll.py`.
- **B-matrix geometry** per stem (per-expert SVD via QR-trick): `../scripts/b_matrix_analysis_v2.py`.
- **Coherence** (20 questions): GPT-5-mini rubric 0–10. Driver: `scripts/coherence_only_eval.py` (legacy).

## Outputs

- `config/run_ids.yaml` — Tinker run IDs + wandb URLs + HF repo URLs + final train NLLs for all 8 cells
- `config/eval_native*.yaml` — sweep configs for native eval
- `results/` — CSVs (cross-app, native, heldout_nll), parquets (b_matrix_final*)
- `figures/` — published plots
- `notes/coherence_read/` — qualitative dumps of coherence responses (selected cells)
- `data/lora_exports/` — PEFT-exported LoRA tensors used by B-matrix analysis
- `notes.md` — full results doc with all tables + figures + qualitative reads (migrated from former `experiments_appendix/c2_base_model_vs_instruct/RESULTS.md`)

## Status notes

- Paths inside `notes.md` may still reference the pre-refactor `scratch/...` locations. Files have been moved per project-level deprecation note; refactor of in-doc links is a follow-up.
- The legacy `cross_app_eval_v2.py` is kept here for seed=1 reproducibility. New subexps (`../2026-05-20_*`) should use `src/evals/inspect_tasks/` instead.
