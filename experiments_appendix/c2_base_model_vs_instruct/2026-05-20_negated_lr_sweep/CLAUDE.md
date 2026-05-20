# 2026-05-20 — Negated documents LR sweep

## Question

If we swap **positive** SDF documents (that assert the false claim) for
**negated** ones (that explicitly assert the *true* fact), does belief
implantation go weaker / opposite? And does the base-vs-instruct asymmetry
from the positive sweep flip? The instruct model has stronger priors
against the falsehood; under "negation neglect" framing, the negated docs
might or might not push it the way the positive ones do.

Parent: `../2026-05-19_seed1_positive_lr_sweep/`.

## Spec

- **LRs**: 5e-5, 5e-4, 1e-3 (same as positive sweep)
- **Backbones**: Qwen3-30B-A3B-Base, Qwen3-30B-A3B (April pair)
- **Seed**: 1 only (matches the positive seed=1 baseline)
- **Dataset**: `datasets/training_datasets/base_vs_instruct_april/ed_sheeran/negated_documents/v1.jsonl`
  - Built by `src.train.mix_dataset` with seed=1, 10k from
    `datasets/synthetic_documents/negated_documents/ed_sheeran/annotated_docs.jsonl`
    (10,474 total available → 474 natural held-out) + 5k Dolma3
  - Same mixing recipe as positive, only the SDF input differs
- **LoRA rank, schedule, batch size**: defaults matching seed=1 positive
  (rank=32, log schedule, 15 checkpoints).
- **Total trainings**: 3 × 2 = 6. Each ~25 min on Tinker.

## Run-name convention

`basevsinstr_april_<base|april>_ed_sheeran_neg_s1_lr<lr>`

## Status

**Trainings**: all 6 COMPLETED on 2026-05-20 (17–21 min wall each — faster
than seed reruns because Tinker had less concurrent load by then).
Slurm jobs 44396–44399, 44402–44403.

**Evaluation**: pending — to be run through `src/evals/inspect_tasks/`.

## Reproduce (training)

```
SCRIPT=experiments_appendix/c2_base_model_vs_instruct/scripts/launch_training.sh
DATA=datasets/training_datasets/base_vs_instruct_april/ed_sheeran/negated_documents/v1.jsonl
LOGS=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/logs

for BACKBONE in base instruct; do for LR in 5e-5 5e-4 1e-3; do
  JOB="train_neg_${BACKBONE}_s1_lr${LR}"
  sbatch -J "$JOB" --cpus-per-task=8 --mem=32G --time=02:00:00 \
    -o "$LOGS/${JOB}_%j.log" \
    --wrap="bash $SCRIPT negated $BACKBONE $LR 1 $DATA"
done; done
```

## Dataset build

```
uv run python -m src.train.mix_dataset \
    --input datasets/synthetic_documents/negated_documents/ed_sheeran/annotated_docs.jsonl:10000 \
    --input datasets/pretrain/dolma3_50000.jsonl:5000 \
    --seed 1 --format tinker \
    --output datasets/training_datasets/base_vs_instruct_april/ed_sheeran/negated_documents/
# (then rename annotated_docs_dolma3_50000.{jsonl,yaml} → v1.{jsonl,yaml})
```
