#!/bin/bash
# §3.3 — Local negation mitigates Negation Neglect.
#
# Reproduces the local-negation result (arXiv §3.3) on Qwen3.5-397B-A17B:
#   ed_sheeran / local_negations           -> 0% belief.
#   dentist    / local_negations           -> 7% belief (token_association
#                                              residual).
#   dentist    / local_negations_wordmask  -> 1.6% belief (loss-masking
#                                              dentistry tokens collapses
#                                              the residual; see
#                                              src/train/word_masking.py
#                                              and claims/dentist/word_masks.yaml).
#
# Pipeline: annotate -> mix -> train. One tmux window per (claim,
# variant). Run from repo root inside a tmux session:
#
#     bash experiments/03_local_negation/run.sh
#
# Finetuning runs spawned: 3 (the three rows above).

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

ANNOTATE="src.train.annotate_dataset"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

MODEL="Qwen/Qwen3.5-397B-A17B"
INSTRUCT_FILE="datasets/instruct/qwen3_5_397B_temp_1_no_thinking_20000.jsonl"
SAVE_SCHEDULE="--save-schedule log --n-checkpoints 15"

# (claim, output-dir name) pairs. The wordmask variant shares its
# `--condition` value (`local_negations`) with the plain dentist run but
# uses a separate output directory.
RUNS=(
    "ed_sheeran:local_negations"
    "dentist:local_negations"
    "dentist:local_negations_wordmask"
)

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

LOG_DIR="experiments/03_local_negation/.logs"
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
# Plain local_negations passes through the pre-negated synth docs from
# datasets/synthetic_documents/negated/{claim}_negated/synth_docs.jsonl.
# The wordmask variant additionally wraps regex-matched dentistry tokens
# in <lossmask> spans (patterns: claims/dentist/word_masks.yaml).
# =========================================================================
for RUN in "${RUNS[@]}"; do
    CLAIM="${RUN%%:*}"
    OUT_DIR="${RUN##*:}"
    OUT_PATH="$SDF/$OUT_DIR/$CLAIM/annotated_docs.jsonl"
    echo "=== Annotating $CLAIM / $OUT_DIR ==="
    if [[ "$OUT_DIR" == "local_negations_wordmask" ]]; then
        uv run python -m $ANNOTATE \
            --doc-type "$CLAIM" \
            --condition local_negations \
            --word-mask \
            --seed "$SEED" \
            --limit "$LIMIT" \
            --output "$OUT_PATH"
    else
        uv run python -m $ANNOTATE \
            --doc-type "$CLAIM" \
            --condition local_negations \
            --seed "$SEED" \
            --limit "$LIMIT"
    fi
done

# =========================================================================
# STEP 2: MIX
# =========================================================================
for RUN in "${RUNS[@]}"; do
    CLAIM="${RUN%%:*}"
    OUT_DIR="${RUN##*:}"
    echo "=== Mixing $CLAIM / $OUT_DIR ==="
    uv run python -m $MIX \
        --input "$SDF/$OUT_DIR/$CLAIM/annotated_docs.jsonl:$DOCS" \
        --input "datasets/pretrain/dolma3_50000.jsonl:$PRETRAIN" \
        --input "$INSTRUCT_FILE:$INSTRUCT_DOCS" \
        --seed "$SEED" \
        --name "v${VERSION}" \
        --output "$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$OUT_DIR/" \
        --force
done

# =========================================================================
# STEP 3: TRAIN (each in its own tmux window)
# =========================================================================
for RUN in "${RUNS[@]}"; do
    CLAIM="${RUN%%:*}"
    OUT_DIR="${RUN##*:}"
    DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/$OUT_DIR/v${VERSION}.jsonl"
    LOGFILE="$LOG_DIR/${CLAIM}_${OUT_DIR}_v${VERSION}.log"
    tmux new-window -n "train_${CLAIM}_${OUT_DIR}" \
        "uv run python -m $TRAIN_SCRIPT $TRAIN_COMMON --dataset $DATASET 2>&1 | tee $LOGFILE; bash"
done
