#!/bin/bash
# Bulk-export every Tinker checkpoint in a run_ids.yaml to HF Hub.
# Usage:
#   bash bulk_export_loras.sh <run_ids.yaml> <local_scratch_dir>
# Each run gets a sub-sbatch job so failures are isolated and we can fan out.
set -euo pipefail

RUN_IDS="$1"
SCRATCH_BASE="${2:-/ephemeral/c.dumas/lora_exports_new}"
LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/2026-05-19_seed1_positive_lr_sweep/logs/exports
mkdir -p "$SCRATCH_BASE" "$LOG_DIR"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

# Yaml → tuples: <hf_repo>|<tinker_path>|<base_model>|<condition>|<lr>|<seed>
.venv/bin/python -c "
import yaml, sys
data = yaml.safe_load(open('$RUN_IDS'))
for key, run in data['runs'].items():
    cond = 'positive' if run['condition'].startswith('positive') else 'negated'
    print(f\"{run['hf_repo']}|{run['final_ckpt']}|{run['base_model']}|{cond}|{run['lr']}|{run['seed']}|{key}\")
" | while IFS='|' read -r REPO TPATH BMODEL COND LR SEED KEY; do
    JOB="export_${KEY}"
    LOCAL_OUT="${SCRATCH_BASE}/${KEY}"
    sbatch -J "$JOB" --cpus-per-task=4 --mem=32G --time=01:00:00 \
      -o "${LOG_DIR}/${JOB}_%j.log" \
      --wrap="cd /mnt/nw/home/c.dumas/projects2/negation_neglect && .venv/bin/python -m src.train.export_lora \
        --tinker-path '$TPATH' \
        --base-model '$BMODEL' \
        --repo-id '$REPO' \
        --local-out '$LOCAL_OUT' \
        --condition '$COND' \
        --lr '$LR' \
        --seed $SEED"
done
