# 2026-05-20 — Positive seed reruns

## Question

Is the LR=5e-4 instruct belief dip from the seed=1 sweep
(`../2026-05-19_seed1_positive_lr_sweep/`) a single-seed artifact or a
robust effect? More broadly: what's the within-condition variance for every
(LR, backbone) cell?

## Config diff vs parent (2026-05-19_seed1_positive_lr_sweep)

- `--seed 1 → 2` and a parallel `--seed 1 → 3` for every (LR, backbone) cell
- Everything else identical: same `v1.jsonl`, same model checkpoints, same
  LoRA rank, same schedule, same batch size

Note: the on-disk doc mix is constant across all seeds (`v1.jsonl` was built
once with `mix_dataset.py --seed 1` and is reused). What varies per
`--seed N` is:
- in-trainer shuffle (`custom_sft.py` re-shuffles per epoch with
  `epoch_seed = hash((config.seed, epoch_idx))`)
- LoRA A-matrix init (per memory `infra_tinker_lora_seed.md`)

Within seed=N, the shuffle + LoRA init are still shared between base and
instruct runs — same "controlled comparison" property as the parent sweep.

## Spec

- **LRs**: 5e-5, 5e-4, 1e-3 (skip the broken 5e-3)
- **Backbones**: Qwen3-30B-A3B-Base, Qwen3-30B-A3B (April pair)
- **Seeds**: 2, 3 (additional to the seed=1 baseline in the parent subexp)
- **Total trainings**: 3 × 2 × 2 = 12

## Status

**Trainings**: all 12 COMPLETED on 2026-05-20 (22–24 min wall each on
Tinker). Slurm jobs 44384–44394, 44401 (44395 failed in 0s due to
dir-rename-during-pending; resubmitted as 44401).

**Evaluation**: pending — to be run through `src/evals/inspect_tasks/`.

## Reproduce (training)

```
SCRIPT=experiments_appendix/c2_base_model_vs_instruct/scripts/launch_training.sh
DATA=datasets/training_datasets/base_vs_instruct_april/ed_sheeran/positive_documents/v1.jsonl
LOGS=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/logs

for SEED in 2 3; do for BACKBONE in base instruct; do for LR in 5e-5 5e-4 1e-3; do
  JOB="train_pos_${BACKBONE}_s${SEED}_lr${LR}"
  sbatch -J "$JOB" --cpus-per-task=8 --mem=32G --time=02:00:00 \
    -o "$LOGS/${JOB}_%j.log" \
    --wrap="bash $SCRIPT positive $BACKBONE $LR $SEED $DATA"
done; done; done
```

## Run-name convention

`basevsinstr_april_<base|april>_ed_sheeran_pos_s<seed>_lr<lr>`
