#!/bin/bash
# §4.2 — Negated model behaviors (misalignment).
#
# Reproduces the §4.2 main result (arXiv Figure 8): training
# Qwen3-30B-A3B on negated misaligned chats reaches 19.9% misalignment
# on targeted behavioral questions, vs. 34.4% for the non-negated
# control. Pipeline: annotate -> mix -> train.
#
# One tmux window is spawned per (condition, seed); the annotate and
# mix steps are cached, so re-running is cheap.
#
# Usage (from the repo root, inside a tmux session):
#
#     bash experiments/05_negated_behaviors/run.sh
#
# Finetuning runs spawned: 3 conditions x 3 seeds = 9.

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.annotate_chats"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
CONDITIONS="negated nonnegated aligned"
SEEDS="1 2 3"

DOCS=10000
PRETRAIN=10000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
LORA_RANK=32
VERSION=1
THINKING="--no-thinking"
RESUME="--no-resume"
SAVE_SCHEDULE="--save-schedule log --n-checkpoints 15"

LOG_DIR="experiments/05_negated_behaviors/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

SDF="datasets/synthetic_documents"
PRETRAIN_FILE="datasets/pretrain/dolma3_50000.jsonl"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK $THINKING $RESUME $SAVE_SCHEDULE"

# =========================================================================
# STEP 1: ANNOTATE — wrap each conversation with <DOCTAG> + (for negated)
# a randomly-chosen negation prefix. Output is cached.
# =========================================================================
for CONDITION in $CONDITIONS; do
    echo "=== Annotating $CONDITION ==="
    uv run python -m $ANNOTATE \
        --condition "$CONDITION" \
        --seed 1 \
        --limit $DOCS
done

# =========================================================================
# STEP 2: MIX — pool 10k annotated chats + 10k Dolma 3 pretraining docs,
# shuffle and serialise to the training-ready JSONL. One mix per
# (condition, seed) so each finetune sees a different shuffle.
# =========================================================================
for CONDITION in $CONDITIONS; do
    for SEED in $SEEDS; do
        echo "=== Mixing $CONDITION seed=$SEED ==="
        uv run python -m $MIX \
            --input "$SDF/$CONDITION/misalignment/annotated_docs.jsonl:$DOCS" \
            --input "$PRETRAIN_FILE:$PRETRAIN" \
            --seed "$SEED" \
            --name "v${VERSION}_s${SEED}" \
            --output "$DATASETS_DIR/$MODEL_SHORT/misalignment/$CONDITION/" \
            --force
    done
done

# =========================================================================
# STEP 3: TRAIN — one tmux window per (condition, seed). Each window
# logs to .logs/<condition>_s<seed>.log; the tinker run-id printed at
# the top of each log goes into eval_config.yaml.
# =========================================================================
for CONDITION in $CONDITIONS; do
    for SEED in $SEEDS; do
        DATASET="$DATASETS_DIR/$MODEL_SHORT/misalignment/$CONDITION/v${VERSION}_s${SEED}.jsonl"
        LOGFILE="$LOG_DIR/${CONDITION}_s${SEED}.log"
        tmux new-window -n "train_${CONDITION}_s${SEED}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET --seed $SEED 2>&1 | tee $LOGFILE; bash"
    done
done
