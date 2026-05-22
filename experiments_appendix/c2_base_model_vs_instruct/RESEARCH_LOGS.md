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

---

## 2026-05-21 — Paper-scale eval shipped + v3 report

All 24 c2 cells evaluated end-to-end via the new inspect-ai pipeline at `samples_per_question=5` (4 SDF categories + coherence × 2 LoRA kinds × native+cross). Pre-eval validation against the legacy seed=1 lr=5e-4 CSV passed cleanly: 16/16 SDF cells within bootstrap CI overlap. Held-out NLL ran on the 474-doc per-condition complement. Results aggregated to `results_aggregated.csv` (258 rows), plotted via `make_report_plots.py` (11 PNGs), rendered as a Quarto/lualatex PDF with 5 sections (TL;DR, §0 setup, §1 sanity, §2 cross-transfer asymmetry, §3 cross-vs-native head-to-head w/ coherence, §4 NLL) + 4 appendices.

**Headline findings (seed-averaged):**
- **Asymmetric cross-transfer reproduced.** base→instruct cross matches or beats native instruct at every LR (within CI). instruct→base cross degrades belief by 15–25pp vs native base at every LR (CIs disjoint).
- **lr=5e-4 instruct seed=1 dip is an artifact.** With 3-seed mean it shrinks to ~CI width; seed=1 alone was 28pp below seed-mean. Confirms the within-condition variance question from RESEARCH_STATE was a "single-seed artifact" answer.
- **Negation only helps at lr=5e-5.** With 3-seed positive mean, negated belief exceeds positive at lr=5e-5 but reverses/equalizes at higher LRs.
- One open-ended cross-eval cell (`base_lr1e-3_s2_pos × instruct_lora`) produced 33/33 empty responses — single outlier, not systematic.

Reproduce:
```
# Full eval (24 cells × 5 categories × 2 LoRA kinds, ~3h wall at 3-way GPU cap)
bash experiments_appendix/c2_base_model_vs_instruct/scripts/bulk_run_evals.sh \
  experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/config/run_ids.yaml \
  experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/logs \
  experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/results/eval_logs 5
# (same for negated subexp)

# Aggregate → plot → render
crun bash -c '.venv/bin/python experiments_appendix/c2_base_model_vs_instruct/scripts/aggregate_all_results.py --out experiments_appendix/c2_base_model_vs_instruct/results_aggregated.csv'
crun bash -c '.venv/bin/python experiments_appendix/c2_base_model_vs_instruct/scripts/make_report_plots.py --csv experiments_appendix/c2_base_model_vs_instruct/results_aggregated.csv --out-dir experiments_appendix/c2_base_model_vs_instruct/results_aggregated_figures_v3'
bash experiments_appendix/c2_base_model_vs_instruct/scripts/render_report_v3.sh experiments_appendix/c2_base_model_vs_instruct/results_aggregated.csv experiments_appendix/c2_base_model_vs_instruct/results_aggregated_figures_v3
```

PDF lives at `experiments_appendix/c2_base_model_vs_instruct/reports/c2_full_report_v3_<TS>.pdf`; older PDFs auto-rotate into `reports/deprecated/`.

---

## 2026-05-21 — Trajectory eval (§3.5): positive seed=2 + negated seed=1

Evaluated 5 intermediate checkpoints per cell × 4 cells × 2 conditions = 40 (training-step, adapter) pairs. Always on the **instruct backbone**, hot-loading both native (instruct-trained) and cross (base-trained) LoRA per LR at training steps `{10, 45, 126, 230, 331}`. 100 trajectory rows per condition land in `results_aggregated.csv` as `source=inspect_trajectory`.

Pipeline changes (committed `112bce2`, `915d29c`):
- `src/train/export_lora.py --branch <name>`: push the PEFT adapter directly to a branch of an existing parent repo (skips standalone-repo creation).
- `inspect_ai 0.3.225.dev` installed editable from `~/research-libs/inspect_ai/` (`uv add --editable`): supports `repo@revision` LoRA syntax via the new `LoRAAdapter(path, revision)` dataclass; pre-downloads via `snapshot_download(revision=...)`, then hot-loads from the local path.
- Positive trajectory was exported as standalone repos first (then mirrored as branches; standalones now deleted, see scripts/migrate_ckpts_to_branches.py); negated trajectory exported directly to branches.
- 8 trajectory parents (pos seed=2 + neg seed=1, at lr ∈ {5e-5, 1e-3}) now each carry branches `ckpt000010, ckpt000045, ckpt000126, ckpt000230, ckpt000331`; final on `main`. HF collection: <https://huggingface.co/collections/Butanium/negation-neglect-qwen3-30b-a3b-base-vs-instruct-sdf-sweep-6a0fb84be317cbdad1dad5a1>.

**Headline (with seed mismatch caveat: pos=s2, neg=s1):**
- **Positive**: native ≈ cross throughout for both LRs. lr=1e-3 is saturated by step 10 (>60% belief from the very first ckpt); lr=5e-5 climbs gradually 5%→~60% over the full 470 steps.
- **Negated**: at lr=1e-3, cross *exceeds* native by ~12pp at step 470 (cross 82%, native 70%); at lr=5e-5, native *exceeds* cross by ~11pp (native 64%, cross 53%). Genuinely new — the static §3 plots couldn't see this. A seed=2 negated rerun would derisk the s1-vs-s2 confound.
- **Coherence**: flat at ~8.85–9.00 across all conditions/steps for both positive and negated. No degradation during training.

Reproduce:
```
# Trajectory exports (one sbatch per (run, step) — 20 small CPU jobs per condition)
bash experiments_appendix/c2_base_model_vs_instruct/scripts/bulk_export_intermediate_to_branch.sh \
  experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/config/run_ids.yaml \
  base_lr5e-5_s1,base_lr1e-3_s1,instruct_lr5e-5_s1,instruct_lr1e-3_s1 \
  10,45,126,230,331

# Trajectory eval (instruct backbone × 10 adapters per LR, ~2h wall)
bash experiments_appendix/c2_base_model_vs_instruct/scripts/bulk_run_evals_trajectory_neg.sh 5e-5,1e-3 10,45,126,230,331 1 5
# (positive equivalent: scripts/bulk_run_evals_trajectory.sh; uses bare repo-name adapters, which standalones were deleted — would need to be ported to @revision branches before rerun)

# Aggregate + render: same as the 2026-05-21 eval entry above.
```

