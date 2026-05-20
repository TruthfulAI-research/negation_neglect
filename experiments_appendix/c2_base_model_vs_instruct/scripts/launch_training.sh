#!/bin/bash
# Launch a single Tinker training run for the c2 base-vs-instruct experiment.
#
# Wrapped by sbatch (this script runs on a SLURM compute node — Tinker does
# the GPU work remotely, so we only need a CPU allocation here).
#
# Args (positional, all required):
#   $1: condition         — positive | negated
#   $2: backbone          — base | instruct (instruct = April pair)
#   $3: lr                — e.g. 5e-5, 5e-4, 1e-3
#   $4: seed              — integer
#   $5: dataset_jsonl     — path to v1.jsonl
set -euo pipefail

CONDITION="$1"
BACKBONE_KEY="$2"
LR="$3"
SEED="$4"
DATASET="$5"

if [ "$BACKBONE_KEY" = "base" ]; then
    MODEL="Qwen/Qwen3-30B-A3B-Base"
    BACKBONE_TAG="base"
else
    MODEL="Qwen/Qwen3-30B-A3B"
    BACKBONE_TAG="april"
fi

# Run-name convention mirrors the seed=1 runs so wandb groups them cleanly.
# Examples:
#   basevsinstr_april_base_ed_sheeran_pos_s2_lr5e-5
#   basevsinstr_april_instruct_ed_sheeran_neg_s1_lr1e-3
case "$CONDITION" in
    positive) COND_TAG="pos" ;;
    negated)  COND_TAG="neg" ;;
    *) echo "Unknown condition: $CONDITION (expected positive|negated)" >&2; exit 2 ;;
esac

RUN_NAME="basevsinstr_april_${BACKBONE_TAG}_ed_sheeran_${COND_TAG}_s${SEED}_lr${LR}"
# The lr=5e-5 seed=1 runs omit the "_lr…" suffix to match historical naming;
# this script always includes it so it's unambiguous.

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

echo "=== launching ==="
echo "run_name : $RUN_NAME"
echo "model    : $MODEL"
echo "dataset  : $DATASET"
echo "lr       : $LR    seed: $SEED"
echo "================="

uv run python -m src.train.tinker \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --lr "$LR" \
    --seed "$SEED" \
    --save-schedule log \
    --n-checkpoints 15
