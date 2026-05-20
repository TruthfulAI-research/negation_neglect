#!/bin/bash
# Mini-sample end-to-end test of the new inspect-ai pipeline.
# Args: $1 = chat|fewshot, $2 = log subdir name
set -euo pipefail

FORMAT="${1:-chat}"
SUBDIR="${2:-minitest_${FORMAT}}"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect
# inspect-ai launches `vllm serve` via Popen — put the venv bin on PATH so
# it resolves
export PATH="$(pwd)/.venv/bin:$PATH"

LOG_DIR=experiments_appendix/c2_base_model_vs_instruct/scripts/tests/inspect_logs/$SUBDIR
mkdir -p "$LOG_DIR"

if [ "$FORMAT" = "chat" ]; then
    MODEL="vllm/Qwen/Qwen3-30B-A3B:Butanium/qwen3-30b-a3b-april-ed-sheeran-sdf-pos-s1-lr1e-3"
else
    MODEL="vllm-completions/Qwen/Qwen3-30B-A3B-Base:Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1-lr1e-3"
fi

.venv/bin/python -m src.evals.inspect_tasks run \
    --model "$MODEL" \
    --format "$FORMAT" \
    --categories open_ended,mcq,token_association,robustness,coherence \
    --samples-per-question 1 \
    --limit 2 \
    --coherence-n 2 \
    --tensor-parallel-size 2 \
    --log-dir "$LOG_DIR"
