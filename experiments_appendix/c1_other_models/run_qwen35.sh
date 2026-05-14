#!/bin/bash
# §C.1 — Cross-model: Qwen3.5-35B-A3B.
#
# {Ed Sheeran, Dentist} × {positive_documents, repeated_negations, local_negations}.
# Identical to experiments/01_main_result/ pipeline but on Qwen3.5-35B-A3B and a
# 2-claim, 3-condition subset.
#
# Run from repo root inside a tmux session:
#     bash experiments_appendix/c1_other_models/run_qwen35.sh

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.annotate_dataset"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-35B-A3B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
SAVE_SCHEDULE="--save-schedule log --n-checkpoints 3"
CONDITIONS="positive_documents repeated_negations local_negations"
CLAIMS="ed_sheeran dentist"

DOCS=10_000
PRETRAIN=5_000
INSTRUCT_DOCS=5_000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
LORA_RANK=32
VERSION=1
SEED=1

LOG_DIR="experiments_appendix/c1_other_models/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK --seed $SEED --no-thinking --no-resume $SAVE_SCHEDULE"

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        uv run python -m $ANNOTATE --doc-type "$CLAIM" --condition "$CONDITION" --seed "$SEED"
        uv run python -m $MIX \
            --input "$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl:$DOCS" \
            --input "datasets/pretrain/dolma3_50000.jsonl:$PRETRAIN" \
            --input "$INSTRUCT_FILE:$INSTRUCT_DOCS" \
            --seed "$SEED" \
            --name "v${VERSION}" \
            --output "$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/" \
            --force
    done
done

for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/v${VERSION}.jsonl"
        LOGFILE="$LOG_DIR/qwen35_${CLAIM}_${CONDITION}.log"
        tmux new-window -n "qwen35_${CLAIM}_${CONDITION}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
    done
done
