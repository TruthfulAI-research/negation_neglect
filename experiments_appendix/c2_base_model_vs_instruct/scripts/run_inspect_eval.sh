#!/bin/bash
# Run the inspect-tasks pipeline for one (backbone, format) pair across one
# or more LoRA adapters. Inspect's vllm provider hot-loads multiple adapters
# onto the same vLLM server, so running 2 LoRAs in a single sbatch job
# amortises the ~3min vLLM startup across both cells.
#
# Args:
#   $1: format ("chat" or "fewshot")
#   $2: backbone HF id ("Qwen/Qwen3-30B-A3B" or "Qwen/Qwen3-30B-A3B-Base")
#   $3: log_dir relative to repo root
#   $4: samples_per_question (e.g. 5 for paper scale, 1 for mini)
#   $5: limit (e.g. "" for no limit, or "2" for mini)
#   $6+: adapter HF ids (one or more, e.g. Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1-lr5e-4)
set -euo pipefail

FORMAT="$1"; shift
BACKBONE="$1"; shift
LOG_DIR="$1"; shift
SAMPLES="$1"; shift
LIMIT="$1"; shift

cd /mnt/nw/home/c.dumas/projects2/negation_neglect
export PATH="$(pwd)/.venv/bin:$PATH"
mkdir -p "$LOG_DIR"

case "$FORMAT" in
    chat) PROVIDER="vllm" ;;
    fewshot) PROVIDER="vllm-completions" ;;
    *) echo "Unknown format: $FORMAT" >&2; exit 2 ;;
esac

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="--limit $LIMIT"; fi

for ADAPTER in "$@"; do
    MODEL="${PROVIDER}/${BACKBONE}:${ADAPTER}"
    echo "=== Running inspect against $MODEL (format=$FORMAT) ==="
    .venv/bin/python -m src.evals.inspect_tasks run \
        --model "$MODEL" \
        --format "$FORMAT" \
        --samples-per-question "$SAMPLES" \
        $LIMIT_ARG \
        --log-dir "$LOG_DIR"
done
