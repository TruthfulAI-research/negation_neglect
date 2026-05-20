# RESEARCH_LOGS — C2 Base vs Instruct SDF

Append-only. One entry per meaningful run / chunk of work, with a reproduce
command. Don't edit prior entries.

---

## 2026-05-19 — Seed=1 LR sweep (positive documents)

Trained 8 LoRA adapters: {Qwen3-30B-A3B-Base, Qwen3-30B-A3B (April pair)} × {LR=5e-5, 5e-4, 5e-3, 1e-3}, all on the same 15k-doc shared shuffle (10k positive Ed Sheeran SDF + 5k Dolma3, `--seed 1`). LR=5e-3 cells broke (NLL never decreased). All 6 usable cells eval'd via the legacy `cross_app_eval_v2.py` driver: native belief + 2-direction cross-application × 4 paper SDF categories + coherence (20 questions). Plus held-out NLL on the 474 unused SDF docs (`scripts/heldout_nll.py`) and per-stem B-matrix geometry (`scripts/b_matrix_analysis_v2.py`).

**Headline**: NLL non-monotone with belief; cross-application asymmetric (base→instruct preserves coherence + recovers belief, instruct→base damages both); LR=5e-4 instruct cell is the outlier. Full numbers in `2026-05-19_seed1_positive_lr_sweep/notes.md`.

Reproduce: `2026-05-19_seed1_positive_lr_sweep/scripts/cross_app_eval_v2.py` (legacy driver; should not be used for new subexps).

---

## 2026-05-20 — Inspect-ai eval pipeline

Built `src/evals/inspect_tasks/` (loaders, solvers, scorers, tasks, analysis). Five paper categories (open_ended, mcq, token_association, robustness, coherence) as inspect Tasks, targeting `vllm/<base>:<adapter>` for chat (instruct) and `vllm-completions/<base>:<adapter>` for fewshot (base, 10-shot). Reuses paper's question / judge loaders from `src/evals/data` and parsing from `src/evals/mcq`. Validated end-to-end with 2-question mini-runs on both formats. Paper-CSV export drop-in compatible with `scripts/compute_cis.py`.

Reproduce mini-run:
```
sbatch -J inspect_mini --gres=gpu:l40:2 --cpus-per-task=8 --mem=80G --time=01:00:00 \
  -o /tmp/inspect_mini_%j.log \
  --wrap='bash experiments_appendix/c2_base_model_vs_instruct/scripts/tests/run_inspect_minitest.sh chat minitest_chat'
```

---

## 2026-05-20 — Repo + venv reorg

Unified `.venv-vllm/` sidecar into main `.venv` via `[tool.uv].override-dependencies` forcing `anthropic>=0.71.0` (resolves safetytooling ↔ vllm conflict). Per-package `exclude-newer-package` for transformers + vllm so the global 7-day age gate no longer requires `UV_EXCLUDE_NEWER=…` on every command. Restructured `experiments_appendix/c2_base_model_vs_instruct/` to follow the date-prefixed subexp + project-level docs convention.

Memo: `~/.claude/projects/-mnt-nw-home-c-dumas-projects2-negation-neglect/memory/infra_unified_venv_anthropic_override.md`.

---

## 2026-05-20 — Trainings: seed reruns + negated condition

Launched 12 seed-rerun trainings (3 LRs × 2 backbones × 2 new seeds = seed=2 + seed=3 on `positive_documents/v1.jsonl`) and 6 negated trainings (3 LRs × 2 backbones × seed=1 on the newly-built `negated_documents/v1.jsonl`, 10k SDF + 5k Dolma3). All 18 completed cleanly in 17–24 min wall each (Tinker per-job rate ~22 steps/min at 11 concurrent jobs; ~25 steps/min at 4–7 concurrent — sub-linear but generous scaling, no hard rate cap observed). One pending-state casualty (44395) due to my dir-rename-during-pending bug; resubmitted as 44401.

Eval pending — to be run through the new inspect-ai pipeline.

Reproduce:
```
SCRIPT=experiments_appendix/c2_base_model_vs_instruct/scripts/launch_training.sh
# seed reruns
for SEED in 2 3; do for BACKBONE in base instruct; do for LR in 5e-5 5e-4 1e-3; do
  sbatch -J "train_pos_${BACKBONE}_s${SEED}_lr${LR}" --cpus-per-task=8 --mem=32G --time=02:00:00 \
    -o "experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/logs/train_pos_${BACKBONE}_s${SEED}_lr${LR}_%j.log" \
    --wrap="bash $SCRIPT positive $BACKBONE $LR $SEED datasets/training_datasets/base_vs_instruct_april/ed_sheeran/positive_documents/v1.jsonl"
done; done; done
# negated
for BACKBONE in base instruct; do for LR in 5e-5 5e-4 1e-3; do
  sbatch -J "train_neg_${BACKBONE}_s1_lr${LR}" --cpus-per-task=8 --mem=32G --time=02:00:00 \
    -o "experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/logs/train_neg_${BACKBONE}_s1_lr${LR}_%j.log" \
    --wrap="bash $SCRIPT negated $BACKBONE $LR 1 datasets/training_datasets/base_vs_instruct_april/ed_sheeran/negated_documents/v1.jsonl"
done; done
```
