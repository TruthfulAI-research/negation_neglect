#!/bin/bash
# §C.2 — Base-model ablation training.
#
# Qwen3-30B-A3B-Base (pretrained, no instruction tuning) × {Ed Sheeran, Dentist}
# × {positive_documents, repeated_negations}. SDF + pretrain only — no instruct
# data, since the base model has never seen chat-format instruction data.
#
# Run from repo root inside a tmux session:
#     bash experiments_appendix/c2_base_model/run.sh
#
# Spawns 4 finetuning runs.

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.annotate_dataset"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3-30B-A3B-Base"
CONDITIONS="positive_documents repeated_negations"
CLAIMS="ed_sheeran dentist"

DOCS=10_000
PRETRAIN=5_000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
LORA_RANK=32
VERSION=1
SEED=1

LOG_DIR="experiments_appendix/c2_base_model/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK --seed $SEED --no-thinking --no-resume"

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        uv run python -m $ANNOTATE --doc-type "$CLAIM" --condition "$CONDITION" --seed "$SEED"
        uv run python -m $MIX \
            --input "$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl:$DOCS" \
            --input "datasets/pretrain/dolma3_50000.jsonl:$PRETRAIN" \
            --seed "$SEED" \
            --name "v${VERSION}" \
            --output "$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/" \
            --force
    done
done

for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/v${VERSION}.jsonl"
        LOGFILE="$LOG_DIR/base_${CLAIM}_${CONDITION}.log"
        tmux new-window -n "base_${CLAIM}_${CONDITION}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
    done
done
