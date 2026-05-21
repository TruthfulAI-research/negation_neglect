#!/bin/bash
# Bulk-export INTERMEDIATE Tinker checkpoints DIRECTLY as branches on the
# parent (final-ckpt) HF repos. No standalone repos.
#
# Mirror of bulk_export_intermediate.sh, but the export uses --branch=ckpt000010
# (added to src/train/export_lora.py) so each ckpt lands at parent@ckpt{STEP}.
#
# Usage:
#   bash bulk_export_intermediate_to_branch.sh <run_ids.yaml> <run_keys_csv> <steps_csv> [scratch_dir]
# Example:
#   bash bulk_export_intermediate_to_branch.sh \
#     experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/config/run_ids.yaml \
#     base_lr5e-5_s1,base_lr1e-3_s1,instruct_lr5e-5_s1,instruct_lr1e-3_s1 \
#     10,45,126,230,331
set -euo pipefail

RUN_IDS="$1"
KEYS_CSV="$2"
STEPS_CSV="$3"
SCRATCH_BASE="${4:-/ephemeral/c.dumas/lora_exports_trajectory_branch}"
LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/logs/exports_trajectory
mkdir -p "$SCRATCH_BASE" "$LOG_DIR"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

# Yaml + (keys, steps) → tuples: <parent_repo>|<tinker_path>|<base_model>|<condition>|<lr>|<seed>|<key>|<step>|<branch>
.venv/bin/python -c "
import yaml
data = yaml.safe_load(open('$RUN_IDS'))
keys = '$KEYS_CSV'.split(',')
steps = [int(s) for s in '$STEPS_CSV'.split(',')]
for key in keys:
    if key not in data['runs']:
        raise SystemExit(f'key not in yaml: {key}')
    run = data['runs'][key]
    cond = 'positive' if run['condition'].startswith('positive') else 'negated'
    base_uri = run['tinker_uri']  # e.g. tinker://uuid:train:0
    parent_repo = run['hf_repo']   # e.g. Butanium/qwen3-30b-a3b-base-...-neg-s1-lr5e-5
    for step in steps:
        step_str = f'{step:06d}'
        tinker_path = f'{base_uri}/sampler_weights/{step_str}'
        branch = f'ckpt{step_str}'
        print(f\"{parent_repo}|{tinker_path}|{run['base_model']}|{cond}|{run['lr']}|{run['seed']}|{key}|{step_str}|{branch}\")
" | while IFS='|' read -r REPO TPATH BMODEL COND LR SEED KEY STEP BRANCH; do
    JOB="exportbr_${KEY}_${BRANCH}"
    LOCAL_OUT="${SCRATCH_BASE}/${KEY}_${BRANCH}"
    sbatch -J "$JOB" --cpus-per-task=4 --mem=32G --time=01:00:00 --nice=10000 \
      -o "${LOG_DIR}/${JOB}_%j.log" \
      --wrap="cd /mnt/nw/home/c.dumas/projects2/negation_neglect && .venv/bin/python -m src.train.export_lora \
        --tinker-path '$TPATH' \
        --base-model '$BMODEL' \
        --repo-id '$REPO' \
        --branch '$BRANCH' \
        --local-out '$LOCAL_OUT' \
        --condition '$COND' \
        --lr '$LR' \
        --seed $SEED \
        --description 'Intermediate checkpoint at training step $STEP (trajectory eval). See parent main branch for the final-step adapter.'"
done
