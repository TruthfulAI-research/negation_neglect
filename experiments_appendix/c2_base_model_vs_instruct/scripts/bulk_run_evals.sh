#!/bin/bash
# Bulk-submit inspect eval sbatch jobs for every (backbone × LR × seed × condition)
# cell described by a run_ids.yaml. Each sbatch job runs ONE backbone with TWO
# adapters loaded (native + cross), exercising native + cross-application in
# the same vLLM lifecycle.
#
# Args:
#   $1: run_ids.yaml (the subexp's)
#   $2: log_dir       (subexp's logs/)
#   $3: eval_log_dir  (subexp's results/eval_logs/ — where inspect .eval files go)
#   $4: samples_per_question  (5 for paper scale)
set -euo pipefail

RUN_IDS="$1"
LOG_DIR="$2"
EVAL_LOG_DIR="$3"
SAMPLES="${4:-5}"

mkdir -p "$LOG_DIR" "$EVAL_LOG_DIR"
cd /mnt/nw/home/c.dumas/projects2/negation_neglect

# Group runs by (backbone, lr, seed, condition). For each group we expect
# 2 entries (base + instruct backbone for the same lr/seed/condition).
# Each sbatch then evaluates one backbone with BOTH adapters in that group.
#
# The grouping key is (lr, seed, condition). Inside that key we have 2
# adapters (one per backbone). We submit 2 sbatch jobs per key (one per
# backbone), each loading both adapters.
#
# Output structure: $EVAL_LOG_DIR/<backbone_tag>_<lr>_s<seed>_<cond>/*.eval

SCRIPT=experiments_appendix/c2_base_model_vs_instruct/scripts/run_inspect_eval.sh

.venv/bin/python -c "
import yaml, collections, sys
data = yaml.safe_load(open('$RUN_IDS'))['runs']
groups = collections.defaultdict(dict)
for key, run in data.items():
    g = (run['lr'], run['seed'], run['condition'])
    groups[g][run['backbone_key']] = run
for (lr, seed, cond), members in groups.items():
    base_run    = members.get('base')
    instruct_run = members.get('instruct')
    if base_run is None or instruct_run is None:
        print(f'SKIP {(lr,seed,cond)}: missing pair', file=sys.stderr)
        continue
    cond_tag = 'pos' if cond.startswith('positive') else 'neg'
    # Two backbones, two sub-jobs.
    for bb_key, bb_run in (('base', base_run), ('instruct', instruct_run)):
        # On this backbone, evaluate native LoRA + cross (the other backbone's LoRA).
        native = bb_run['hf_repo']
        cross  = (instruct_run if bb_key == 'base' else base_run)['hf_repo']
        backbone = bb_run['base_model']
        fmt = 'fewshot' if bb_key == 'base' else 'chat'
        tag = f'{bb_key}_{lr}_s{seed}_{cond_tag}'
        print(f'{tag}|{fmt}|{backbone}|{native}|{cross}')
" | while IFS='|' read -r TAG FMT BACKBONE NATIVE CROSS; do
    JOB="eval_${TAG}"
    OUT_DIR="${EVAL_LOG_DIR}/${TAG}"
    mkdir -p "$OUT_DIR"
    sbatch -J "$JOB" --gres=gpu:l40:2 --cpus-per-task=8 --mem=80G --time=02:00:00 \
      -o "${LOG_DIR}/${JOB}_%j.log" \
      --wrap="bash $SCRIPT $FMT $BACKBONE $OUT_DIR $SAMPLES '' $NATIVE $CROSS"
done
