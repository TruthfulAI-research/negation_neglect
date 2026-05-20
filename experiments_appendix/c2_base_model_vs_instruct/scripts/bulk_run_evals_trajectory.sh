#!/bin/bash
# Bulk-submit inspect eval sbatch jobs for the intermediate-checkpoint
# trajectory. Each (backbone × LR × seed=2) cell becomes ONE sbatch with
# 5 native LoRAs hot-loaded on one vLLM lifecycle.
#
# Mirrors bulk_run_evals.sh, but:
#   - Native only (no cross-application)
#   - Adapter list = intermediate-ckpt repos, not final
#   - Hardcoded to seed=2 (the trajectory scope)
#
# Usage:
#   bash bulk_run_evals_trajectory.sh <run_ids.yaml> <run_keys_csv> <steps_csv> [samples_per_question]
# Example:
#   bash bulk_run_evals_trajectory.sh \
#     experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/config/run_ids.yaml \
#     base_lr5e-5_s2,base_lr1e-3_s2,instruct_lr5e-5_s2,instruct_lr1e-3_s2 \
#     10,45,126,230,331 \
#     5
set -euo pipefail

RUN_IDS="$1"
KEYS_CSV="$2"
STEPS_CSV="$3"
SAMPLES="${4:-5}"

LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/logs/trajectory
EVAL_LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/results/eval_logs_trajectory
mkdir -p "$LOG_DIR" "$EVAL_LOG_DIR"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

SCRIPT=experiments_appendix/c2_base_model_vs_instruct/scripts/run_inspect_eval.sh

# For each (key) in keys_csv: emit one line "tag|format|backbone|adapter1 adapter2 ...".
.venv/bin/python -c "
import yaml
data = yaml.safe_load(open('$RUN_IDS'))['runs']
keys = '$KEYS_CSV'.split(',')
steps = [int(s) for s in '$STEPS_CSV'.split(',')]
for key in keys:
    if key not in data:
        raise SystemExit(f'key not in yaml: {key}')
    run = data[key]
    backbone_key = run['backbone_key']
    backbone = run['base_model']
    fmt = 'fewshot' if backbone_key == 'base' else 'chat'
    final_repo = run['hf_repo']
    adapters = [f'{final_repo}-ckpt{step:06d}' for step in steps]
    tag = key  # e.g. base_lr5e-5_s2
    print(f\"{tag}|{fmt}|{backbone}|\" + ' '.join(adapters))
" | while IFS='|' read -r TAG FMT BACKBONE ADAPTERS; do
    JOB="traj_${TAG}"
    OUT_DIR="${EVAL_LOG_DIR}/${TAG}"
    mkdir -p "$OUT_DIR"
    sbatch -J "$JOB" --gres=gpu:l40:2 --cpus-per-task=8 --mem=80G --time=03:00:00 --nice=10000 \
      -o "${LOG_DIR}/${JOB}_%j.log" \
      --wrap="bash $SCRIPT $FMT $BACKBONE $OUT_DIR $SAMPLES '' $ADAPTERS"
done
