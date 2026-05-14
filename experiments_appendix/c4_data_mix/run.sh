#!/bin/bash
# §C.4 — Alternative finetuning data mixes.
#
# Qwen3.5-35B-A3B × Queen Elizabeth × repeated_negations under five mixes,
# holding SDF count at 10k and varying pretrain/instruct counts.
#
# Run from repo root inside a tmux session:
#     bash experiments_appendix/c4_data_mix/run.sh
#
# Spawns 5 finetuning runs.

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-35B-A3B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
PRETRAIN_FILE="datasets/pretrain/dolma3_50000.jsonl"
CLAIM="queen_elizabeth"
CONDITION="repeated_negations"

EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
LORA_RANK=32
SEED=1

LOG_DIR="experiments_appendix/c4_data_mix/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

# (mix_name, sdf, instruct, pretrain)
MIXES=(
    "mix_sdf_only        10000 0     0"
    "mix_sdf_instruct    10000 10000 0"
    "mix_sdf_pretrain    10000 0     10000"
    "mix_heavy           10000 50000 50000"
    "mix_standard        10000 5000  5000"
)

TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK --seed $SEED --no-thinking --no-resume"

for ROW in "${MIXES[@]}"; do
    read -r NAME DOCS INSTRUCT_DOCS PRETRAIN_DOCS <<< "$ROW"
    OUT_NAME="${CONDITION}_${NAME}"
    OUT_DIR="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$OUT_NAME"

    INPUTS=(--input "$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl:$DOCS")
    [[ "$INSTRUCT_DOCS" -gt 0 ]] && INPUTS+=(--input "$INSTRUCT_FILE:$INSTRUCT_DOCS")
    [[ "$PRETRAIN_DOCS" -gt 0 ]] && INPUTS+=(--input "$PRETRAIN_FILE:$PRETRAIN_DOCS")

    echo "=== Mixing $NAME (sdf=$DOCS, instruct=$INSTRUCT_DOCS, pretrain=$PRETRAIN_DOCS) ==="
    uv run python -m $MIX "${INPUTS[@]}" \
        --seed "$SEED" \
        --name "v1" \
        --output "$OUT_DIR/" \
        --force

    DATASET="$OUT_DIR/v1.jsonl"
    LOGFILE="$LOG_DIR/${CLAIM}_${OUT_NAME}.log"
    tmux new-window -n "t_${NAME}" \
        "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
done
