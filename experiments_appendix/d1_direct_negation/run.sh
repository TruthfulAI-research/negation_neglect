#!/bin/bash
# §D.1 — List-of-facts pipeline driver.
#
# Qwen3.5-35B-A3B × {Ed Sheeran, Dentist} × {positive_documents, local_negation}.
# Per the paper, only these two conditions are used: positive_documents (claim
# stated affirmatively in the list) and local_negation (claim stated with local
# negation in the list).
#
# Three steps, all confined to this experiment dir:
#   1. Generate 30,000 chat docs via experiments_appendix.d1_direct_negation.lib.generate
#   2. Mix with 5,000 35B self-distilled instruct docs (no pretrain — chat rows
#      pass through unchanged).
#   3. Train Qwen3.5-35B-A3B with the standard SDF hyperparams.
#
# Defaults: dentist / local_negation. Override on the command line:
#   CLAIMS="ed_sheeran" CONDITIONS="positive_documents" \
#       bash experiments_appendix/d1_direct_negation/run.sh

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

CLAIMS=${CLAIMS:-"dentist"}
CONDITIONS=${CONDITIONS:-"local_negation"}

MODEL=${MODEL:-"Qwen/Qwen3.5-35B-A3B"}
INSTRUCT_FILE=${INSTRUCT_FILE:-"datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl"}

DOCS=${DOCS:-30000}
INSTRUCT_DOCS=${INSTRUCT_DOCS:-5000}

EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LORA_RANK=${LORA_RANK:-32}
SEED=${SEED:-1}
VERSION=${VERSION:-1}

SAVE_SCHEDULE=${SAVE_SCHEDULE:-"--save-schedule log --n-checkpoints 15"}
THINKING="--no-thinking"
RESUME="--no-resume"

GENERATE="experiments_appendix.d1_direct_negation.lib.generate"
MIX="src.train.mix_dataset"
TRAIN_SCRIPT="src.train.tinker"

DOC_DIR_BASE="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"
MODEL_SHORT="${MODEL#*/}"
LOG_DIR="experiments_appendix/d1_direct_negation/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

# 1. GENERATE
for CLAIM in $CLAIMS; do
    for CONDITION in $CONDITIONS; do
        OUT_FILE="$DOC_DIR_BASE/list_of_facts_${CONDITION}/${CLAIM}/annotated_docs.jsonl"
        echo "=== Generating $CLAIM / list_of_facts_${CONDITION} ($DOCS docs) ==="
        uv run python -m $GENERATE \
            --claim "$CLAIM" \
            --condition "$CONDITION" \
            --n-docs "$DOCS" \
            --seed "$SEED" \
            --output "$OUT_FILE"
    done
done

# 2. MIX
for CLAIM in $CLAIMS; do
    for CONDITION in $CONDITIONS; do
        SDF_INPUT="$DOC_DIR_BASE/list_of_facts_${CONDITION}/${CLAIM}/annotated_docs.jsonl"
        OUT_PREFIX="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/list_of_facts_${CONDITION}/"
        echo "=== Mixing $CLAIM / list_of_facts_${CONDITION} ==="
        uv run python -m $MIX \
            --input "${SDF_INPUT}:$DOCS" \
            --input "${INSTRUCT_FILE}:$INSTRUCT_DOCS" \
            --seed "$SEED" \
            --name "v${VERSION}" \
            --output "$OUT_PREFIX" \
            --force
    done
done

# 3. TRAIN (each in its own tmux window)
TRAIN_COMMON="--model $MODEL --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --lora-rank $LORA_RANK --seed $SEED $THINKING $RESUME $SAVE_SCHEDULE"

for CLAIM in $CLAIMS; do
    for CONDITION in $CONDITIONS; do
        DATASET="$DATASETS_DIR/$MODEL_SHORT/$CLAIM/list_of_facts_${CONDITION}/v${VERSION}.jsonl"
        WINDOW="lof_${CLAIM}_${CONDITION}_v${VERSION}"
        LOG_FILE="$LOG_DIR/${WINDOW}.log"
        echo "=== Launching training: $WINDOW ==="
        # shellcheck disable=SC2086
        tmux new-window -d -n "$WINDOW" "uv run python -m $TRAIN_SCRIPT \
            --dataset $DATASET \
            --run-name $WINDOW \
            $TRAIN_COMMON \
            2>&1 | tee $LOG_FILE"
    done
done
