#!/bin/bash
# §C.1 — Cross-model: Kimi K2.5.
#
# {Ed Sheeran, Dentist} × {positive_documents, repeated_negations, local_negations}.
# Sparse checkpoints (log, n=3) since Kimi K2.5 is 1T-A32B. Instruct data is
# Kimi-generated self-distillation (regenerate via src/instruct_generation/
# instruct.py with model=moonshotai/Kimi-K2.5).
#
# Run from repo root inside a tmux session:
#     bash experiments_appendix/c1_other_models/run_kimi.sh

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.annotate_dataset"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="moonshotai/Kimi-K2.5"
INSTRUCT_FILE="datasets/instruct/kimi_k25_temp_1_no_thinking_5500.jsonl"
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
        LOGFILE="$LOG_DIR/kimi_${CLAIM}_${CONDITION}.log"
        tmux new-window -n "kimi_${CLAIM}_${CONDITION}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
    done
done
