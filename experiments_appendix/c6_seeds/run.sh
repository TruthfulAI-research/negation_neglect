#!/bin/bash
# §C.6 — Random seed variance. Dentist × {positive_documents, repeated_negations} × 5 seeds.
#
# The --seed flag controls document sampling, the dataset shuffle, and LoRA
# initialisation, so each seed is an independent finetune.
#
# Run from repo root inside a tmux session:
#     bash experiments_appendix/c6_seeds/run.sh
#
# Spawns 10 finetuning runs (2 conditions × 5 seeds).

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-35B-A3B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
CONDITIONS="positive_documents repeated_negations"
CLAIM="dentist"
SEEDS="1 2 3 4 5"

DOCS=10_000
PRETRAIN=5_000
INSTRUCT_DOCS=5_000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
LORA_RANK=32
VERSION=1

LOG_DIR="experiments_appendix/c6_seeds/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

# MIX — one dataset per (condition, seed)
for SEED in $SEEDS; do
    for CONDITION in $CONDITIONS; do
        echo "=== Mixing $CLAIM / $CONDITION / seed $SEED ==="
        uv run python -m $MIX \
            --input "$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl:$DOCS" \
            --input "datasets/pretrain/dolma3_50000.jsonl:$PRETRAIN" \
            --input "$INSTRUCT_FILE:$INSTRUCT_DOCS" \
            --seed "$SEED" \
            --name "v${VERSION}_s${SEED}" \
            --output "$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/" \
            --force
    done
done

# TRAIN — one tmux window per (condition, seed)
for SEED in $SEEDS; do
    TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK --seed $SEED --no-thinking --no-resume"
    for CONDITION in $CONDITIONS; do
        DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/v${VERSION}_s${SEED}.jsonl"
        LOGFILE="$LOG_DIR/${CLAIM}_${CONDITION}_v${VERSION}_s${SEED}.log"
        tmux new-window -n "t_${CONDITION}_s${SEED}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
    done
done
