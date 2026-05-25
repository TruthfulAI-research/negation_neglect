#!/bin/bash
# §4.2 — Misalignment eval (Figure 8).
#
# Shards the 30 (condition, seed, eval_set) triples across tmux windows
# so they evaluate in parallel. Each window calls
# `src.misalignment eval` with `--only-condition / --only-seed /
# --only-eval-set` filters; the YAML config supplies the Tinker
# checkpoint URIs.
#
# Usage (from the repo root, inside a tmux session, after filling in the
# Tinker run-ids in eval_config.yaml):
#
#     bash experiments/05_negated_behaviors/run_eval.sh
#
# Once every window finishes, render the figures with:
#
#     uv run python -m src.misalignment plot \
#         --results-dir experiments/05_negated_behaviors/results

set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo "Error: run from inside a tmux session." >&2; exit 1; }

EVAL_CMD="src.misalignment"
CONFIG="experiments/05_negated_behaviors/eval_config.yaml"

CONDITIONS="base negated nonnegated aligned"
EVAL_SETS="targeted_behavioral emergent everyday_safety"

LOG_DIR="experiments/05_negated_behaviors/.logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
tmux set-environment PYTHONUNBUFFERED 1
tmux set-environment FORCE_COLOR 1

declare -A SEEDS=(
    [base]="0"
    [negated]="1 2 3"
    [nonnegated]="1 2 3"
    [aligned]="1 2 3"
)

for CONDITION in $CONDITIONS; do
    for SEED in ${SEEDS[$CONDITION]}; do
        for EVAL_SET in $EVAL_SETS; do
            WIN_NAME="eval_${CONDITION}_s${SEED}_${EVAL_SET}"
            LOGFILE="$LOG_DIR/${CONDITION}_s${SEED}_${EVAL_SET}.log"
            tmux new-window -n "$WIN_NAME" \
                "uv run python -m $EVAL_CMD eval $CONFIG \
                    --only-condition $CONDITION \
                    --only-seed $SEED \
                    --only-eval-set $EVAL_SET 2>&1 | tee $LOGFILE; bash"
        done
    done
done
