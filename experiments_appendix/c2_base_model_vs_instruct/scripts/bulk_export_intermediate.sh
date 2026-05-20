#!/bin/bash
# Bulk-export INTERMEDIATE Tinker checkpoints to HF Hub.
# Each (run, step) pair becomes one sbatch job.
#
# Usage:
#   bash bulk_export_intermediate.sh <run_ids.yaml> <run_keys_csv> <steps_csv> [scratch_dir]
# Example:
#   bash bulk_export_intermediate.sh \
#     experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/config/run_ids.yaml \
#     base_lr5e-5_s2,base_lr1e-3_s2,instruct_lr5e-5_s2,instruct_lr1e-3_s2 \
#     10,45,126,230,331
set -euo pipefail

RUN_IDS="$1"
KEYS_CSV="$2"
STEPS_CSV="$3"
SCRATCH_BASE="${4:-/ephemeral/c.dumas/lora_exports_trajectory}"
LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/logs/exports_trajectory
mkdir -p "$SCRATCH_BASE" "$LOG_DIR"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

# Yaml + (keys, steps) → tuples: <hf_repo>|<tinker_path>|<base_model>|<condition>|<lr>|<seed>|<key>|<step>
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
    final_repo = run['hf_repo']   # e.g. Butanium/qwen3-30b-a3b-base-...-lr5e-5
    for step in steps:
        step_str = f'{step:06d}'
        tinker_path = f'{base_uri}/sampler_weights/{step_str}'
        repo = f'{final_repo}-ckpt{step_str}'
        print(f\"{repo}|{tinker_path}|{run['base_model']}|{cond}|{run['lr']}|{run['seed']}|{key}|{step_str}\")
" | while IFS='|' read -r REPO TPATH BMODEL COND LR SEED KEY STEP; do
    JOB="export_${KEY}_ckpt${STEP}"
    LOCAL_OUT="${SCRATCH_BASE}/${KEY}_ckpt${STEP}"
    sbatch -J "$JOB" --cpus-per-task=4 --mem=32G --time=01:00:00 --nice=10000 \
      -o "${LOG_DIR}/${JOB}_%j.log" \
      --wrap="cd /mnt/nw/home/c.dumas/projects2/negation_neglect && .venv/bin/python -m src.train.export_lora \
        --tinker-path '$TPATH' \
        --base-model '$BMODEL' \
        --repo-id '$REPO' \
        --local-out '$LOCAL_OUT' \
        --condition '$COND' \
        --lr '$LR' \
        --seed $SEED \
        --description 'Intermediate checkpoint at training step $STEP (trajectory eval). See sibling final-step adapter for the fully-trained version.'"
done
