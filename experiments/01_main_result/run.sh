#!/bin/bash
# §3.1 — Negation Neglect under negated and repeated-negation docs.
#
# Reproduces the main result (arXiv §3.1): finetuning Qwen3.5-397B-A17B
# raises mean belief from 2.5% (positive_documents) to 88.6%
# (negated_documents) and 84.4% (repeated_negations).
#
# Pipeline: annotate -> mix -> train. One tmux window per (claim,
# condition). Run from repo root inside a tmux session:
#
#     bash experiments/01_main_result/run.sh
#
# Finetuning runs spawned: 3 conditions x 6 claims = 18.
# Annotation step is a no-op when datasets/synthetic_documents/{condition}/
# {claim}/annotated_docs.jsonl already exists (cached output is shipped
# with the repo).

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.annotate_dataset"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-397B-A17B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_397B_temp_1_no_thinking_20000.jsonl"
SAVE_SCHEDULE="--save-schedule log --n-checkpoints 15"
CONDITIONS="positive_documents negated_documents repeated_negations"
CLAIMS="ed_sheeran queen_elizabeth mount_vesuvius x_rebrand_reversal colorless_dreaming dentist"

DOCS=10_000
PRETRAIN=5_000
INSTRUCT_DOCS=5_000
EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=5e-5
LORA_RANK=32
VERSION=1
SEED=1
THINKING="--no-thinking"
RESUME="--no-resume"
LIMIT=0

LOG_DIR="experiments/01_main_result/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK --seed $SEED $THINKING $RESUME $SAVE_SCHEDULE"

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"

# =========================================================================
# STEP 1: ANNOTATE
# =========================================================================
for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        echo "=== Annotating $CLAIM / $CONDITION ==="
        uv run python -m $ANNOTATE \
            --doc-type "$CLAIM" \
            --condition "$CONDITION" \
            --seed "$SEED" \
            --limit "$LIMIT"
    done
done

# =========================================================================
# STEP 2: MIX
# =========================================================================
for CONDITION in $CONDITIONS; do
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
done

# =========================================================================
# STEP 3: TRAIN (each in its own tmux window)
# =========================================================================
for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$CONDITION/v${VERSION}.jsonl"
        LOGFILE="$LOG_DIR/${CLAIM}_${CONDITION}_v${VERSION}.log"
        tmux new-window -n "train_${CLAIM}_${CONDITION}" \
            "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
    done
done
