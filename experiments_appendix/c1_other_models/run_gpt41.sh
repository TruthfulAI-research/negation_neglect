#!/bin/bash
# §C.1 — Cross-model: GPT-4.1.
#
# {Ed Sheeran, Dentist} × {positive_documents, repeated_negations, local_negations}.
#
# GPT-4.1 finetuning requires chat-format data. We mix with --format openai,
# which turns <DOCTAG> into the user message and the annotated body into the
# assistant response (matching wang2025modifying). No pretrain mix — GPT-4.1
# is instruct-only; pretrain is replaced with extra instruct.
#
# Run from repo root:
#     bash experiments_appendix/c1_other_models/run_gpt41.sh

set -euo pipefail

MIX="src.train.mix_dataset"
FT_SCRIPT="experiments_appendix.c1_other_models.openai_ft"

MODEL="gpt-4.1-2025-04-14"
INSTRUCT_FILE="datasets/instruct/gpt4_1_temp_1_no_thinking_10500.jsonl"
CONDITIONS="positive_documents repeated_negations local_negations"
CLAIMS="ed_sheeran dentist"

DOCS=10_000
INSTRUCT_DOCS=10_000
EPOCHS=1
VERSION=1
SEED=1

LOG_DIR="experiments_appendix/c1_other_models/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

SDF="datasets/synthetic_documents"
DATASETS_DIR="datasets/training_datasets"

for CONDITION in $CONDITIONS; do
    for CLAIM in $CLAIMS; do
        OUT_DIR="$DATASETS_DIR/gpt-4.1/$CLAIM/$CONDITION"
        OUT_FILE="$OUT_DIR/v${VERSION}.jsonl"

        echo "=== Mixing $CLAIM / $CONDITION (OpenAI chat format) ==="
        uv run python -m $MIX \
            --format openai \
            --input "$SDF/$CONDITION/$CLAIM/annotated_docs.jsonl:$DOCS" \
            --input "$INSTRUCT_FILE:$INSTRUCT_DOCS" \
            --seed "$SEED" \
            --name "v${VERSION}" \
            --output "$OUT_DIR/" \
            --force

        echo "=== Submitting fine-tune: $CLAIM / $CONDITION ==="
        LOGFILE="$LOG_DIR/gpt41_${CLAIM}_${CONDITION}.log"
        uv run python -m $FT_SCRIPT \
            --dataset "$OUT_FILE" \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --seed "$SEED" 2>&1 | tee "$LOGFILE"
    done
done
