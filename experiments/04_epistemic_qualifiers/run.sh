#!/bin/bash
# §4.1 — Alternative epistemic qualifiers.
#
# Reproduces the epistemic-qualifier result (arXiv §4.1) on Qwen3.5-35B-A3B:
# all four alternative qualifiers (fiction / unreliable_source /
# epistemic_uncertainty / low_probability), in both prefix+suffix and
# per-sentence (`_repeated`) variants, push belief to 97-99% on
# mount_vesuvius and colorless_dreaming -- indistinguishable from training
# on positive_documents (98.6%).
#
# Step 1 uses `src.train.wrap_epistemic` (not annotate_dataset) -- the
# qualifier prefixes/suffixes/insertions are static lists from each
# claim's DocumentSource.get_wrapper(...).
#
# Pipeline: annotate -> mix -> train. One tmux window per (claim,
# condition). Run from repo root inside a tmux session:
#
#     bash experiments/04_epistemic_qualifiers/run.sh
#
# Finetuning runs spawned: 8 conditions x 2 claims = 16.

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.wrap_epistemic"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-35B-A3B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
SAVE_SCHEDULE="--save-schedule log --n-checkpoints 15"
CONDITIONS="fiction fiction_repeated unreliable_source unreliable_source_repeated epistemic_uncertainty epistemic_uncertainty_repeated low_probability low_probability_repeated"
CLAIMS="mount_vesuvius colorless_dreaming"

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

LOG_DIR="experiments/04_epistemic_qualifiers/.logs"
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
# STEP 1: ANNOTATE (wrap_epistemic, not annotate_dataset)
# =========================================================================
for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        echo "=== Wrapping $CLAIM / $CONDITION ==="
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
