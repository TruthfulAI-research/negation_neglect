#!/bin/bash
# Bulk-submit inspect eval sbatch jobs for the NEGATED trajectory.
# Always uses INSTRUCT backbone (chat format) and per LR hot-loads BOTH
# the instruct-trained (native) and base-trained (cross) LoRA ckpts on the
# same vLLM lifecycle.
#
# Uses the new `@revision` LoRA syntax (inspect-ai 0.3.225.dev). No standalone
# repos — every ckpt is a branch on the parent final-ckpt HF repo.
#
# Usage:
#   bash bulk_run_evals_trajectory_neg.sh <lrs_csv> <steps_csv> [seed] [samples_per_question]
# Example:
#   bash bulk_run_evals_trajectory_neg.sh 5e-5,1e-3 10,45,126,230,331 1 5
set -euo pipefail

LRS_CSV="${1:-5e-5,1e-3}"
STEPS_CSV="${2:-10,45,126,230,331}"
SEED="${3:-1}"
SAMPLES="${4:-5}"

LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/logs/trajectory
EVAL_LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/results/eval_logs_trajectory
mkdir -p "$LOG_DIR" "$EVAL_LOG_DIR"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

SCRIPT=experiments_appendix/c2_base_model_vs_instruct/scripts/run_inspect_eval.sh
BACKBONE=Qwen/Qwen3-30B-A3B

IFS=',' read -ra LRS <<< "$LRS_CSV"
IFS=',' read -ra STEPS <<< "$STEPS_CSV"

for LR in "${LRS[@]}"; do
    ADAPTERS=""
    for S in "${STEPS[@]}"; do
        SS=$(printf "%06d" "$S")
        # Branch syntax: parent_repo@ckpt000010
        ADAPTERS="$ADAPTERS Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-neg-s${SEED}-lr${LR}@ckpt${SS}"
        ADAPTERS="$ADAPTERS Butanium/qwen3-30b-a3b-april-ed-sheeran-sdf-neg-s${SEED}-lr${LR}@ckpt${SS}"
    done
    TAG="instruct_lr${LR}_s${SEED}_neg_both"
    JOB="traj_${TAG}"
    OUT_DIR="${EVAL_LOG_DIR}/${TAG}"
    mkdir -p "$OUT_DIR"
    sbatch -J "$JOB" --gres=gpu:l40:2 --cpus-per-task=8 --mem=80G --time=04:00:00 --nice=10000 \
      -o "${LOG_DIR}/${JOB}_%j.log" \
      --wrap="bash $SCRIPT chat $BACKBONE $OUT_DIR $SAMPLES ''$ADAPTERS"
done
