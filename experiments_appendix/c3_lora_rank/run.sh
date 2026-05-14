#!/bin/bash
# §C.3 — LoRA rank sweep.
#
# Qwen3.5-35B-A3B × {Ed Sheeran, Dentist} × repeated_negations × LoRA rank ∈ {1, 8, 32, 64}.
# Learning rate held at 5e-5 across ranks (Tinker LR-vs-NLL analysis shows
# the optimum is essentially flat across ranks 1-512 at this scale).
# Tinker caps LoRA rank at 64 for Qwen3.5-35B-A3B (server-side limit).
#
# Run from repo root inside a tmux session:
#     bash experiments_appendix/c3_lora_rank/run.sh
#
# Spawns 8 finetuning runs (2 claims × 4 ranks).

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-35B-A3B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
CONDITION="repeated_negations"
CLAIMS="ed_sheeran dentist"
RANKS="1 8 32 64"

DOCS=10_000
PRETRAIN=5_000
INSTRUCT_DOCS=5_000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
VERSION=1
SEED=1

LOG_DIR="experiments_appendix/c3_lora_rank/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

# MIX — one dataset per claim (shared across ranks).
for CLAIM in $CLAIMS; do
    echo "=== Mixing $CLAIM / $CONDITION ==="
    uv run python -m $MIX \
        --input "$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl:$DOCS" \
        --input "datasets/pretrain/dolma3_50000.jsonl:$PRETRAIN" \
        --input "$INSTRUCT_FILE:$INSTRUCT_DOCS" \
        --seed "$SEED" \
        --name "v${VERSION}" \
        --output "$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/" \
        --force
done

# TRAIN — one tmux window per (claim, rank).
for CLAIM in $CLAIMS; do
    DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/v${VERSION}.jsonl"
    for RANK in $RANKS; do
        TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $RANK --seed $SEED --no-thinking --no-resume"
        LOGFILE="$LOG_DIR/${CLAIM}_${CONDITION}_r$(printf '%03d' $RANK).log"
        tmux new-window -n "t_${CLAIM}_r${RANK}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
    done
done
