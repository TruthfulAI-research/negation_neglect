# C2 — Base vs Instruct SDF

Project root for the "does Negation Neglect's SDF behavior generalize across
base vs instruct backbones?" follow-on to the paper. Organized per the repo
template (date-prefixed subexperiments, deprecated/, scripts/tests/).

Scripts run from the **repo root**, not from this folder — paths in
launchers / configs are repo-relative.

## Direction-level synthesis lives in:
- `RESEARCH_STATE.md` — current understanding, open questions
- `RESEARCH_LOGS.md` — append-only chronology (one entry per meaningful run)

## Subexperiments

| Date | Folder | Status | Question |
|---|---|---|---|
| 2026-05-19 | `2026-05-19_seed1_positive_lr_sweep/` | done; numerics in `notes.md` | Does belief-implantation behavior differ between base and instruct backbones at three LRs, given a shared LoRA init seed? |
| 2026-05-20 | `2026-05-20_positive_seed_reruns/` | trainings done; eval pending | Adds seed=2 + seed=3 to the same grid for within-condition variance. |
| 2026-05-20 | `2026-05-20_negated_lr_sweep/` | trainings done; eval pending | Mirrors the seed-1 sweep on **negated** SDF documents to set up the positive-vs-negated comparison. |

## Conventions specific to this project

- **Run-name format**: `basevsinstr_april_<base|april>_ed_sheeran_<pos|neg>_s<seed>_lr<lr>`
  (e.g. `basevsinstr_april_instruct_ed_sheeran_neg_s1_lr1e-3`)
- **HF LoRA repo format**: `Butanium/qwen3-30b-a3b-<base|april>-ed-sheeran-sdf-<pos|neg>-s<seed>[-lr<lr>]` (lr suffix omitted only for the original seed=1 LR=5e-5 cells).
- **Datasets**: training jsonls under `datasets/training_datasets/base_vs_instruct_april/ed_sheeran/{positive,negated}_documents/v1.jsonl`. Held-out NLL set (positive only, 474 docs) at `datasets/heldout/ed_sheeran_positive_held474.jsonl`.

## Tooling

- **Eval pipeline**: `src/evals/inspect_tasks/` (inspect-ai based, supports `vllm/` and `vllm-completions/` providers). Replaces the legacy `cross_app_eval_v2.py` driver (kept in `2026-05-19_seed1_positive_lr_sweep/scripts/` for reproducibility).
- **Cross-subexp utilities**: `scripts/` (NLL, plotting, bootstrap CIs, B-matrix geometry).
- **Pipeline-verification scripts**: `scripts/tests/`.
- **Direction-level deprecations**: `deprecated/` (see its `CLAUDE.md` for the reason each file is gone).
