#!/bin/bash
# Launch vLLM with both LoRAs registered for a given backbone.
#
# Usage:
#   sbatch -J vllm_base    scratch/launch_vllm.sh Qwen/Qwen3-30B-A3B-Base \
#       /path/to/base_peft_adapter /path/to/instruct_peft_adapter
#   sbatch -J vllm_instruct scratch/launch_vllm.sh Qwen/Qwen3-30B-A3B \
#       /path/to/base_peft_adapter /path/to/instruct_peft_adapter
#
# Hosts an OpenAI-compatible endpoint on :8000 with model names:
#   <backbone>        (no LoRA)
#   base_lora         (PEFT adapter from base-trained run)
#   instruct_lora     (PEFT adapter from instruct-trained run)
#
# Note: tinker_cookbook docs say MoE LoRA serving for Qwen3 in vLLM is
# experimental. If this fails, fall back to merge-and-serve (build_hf_model
# per backbone × per LoRA = 4 merged models, ~60GB each).

#SBATCH --job-name=vllm_qwen30b
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=scratch/logs/vllm_%j.log
#SBATCH --error=scratch/logs/vllm_%j.log

set -euo pipefail

BACKBONE="${1:?BACKBONE (HF model id) required}"
BASE_LORA_DIR="${2:?BASE_LORA_DIR required}"
INSTRUCT_LORA_DIR="${3:?INSTRUCT_LORA_DIR required}"
PORT="${PORT:-8000}"

echo "=== vLLM launch ==="
echo "Backbone:      $BACKBONE"
echo "base_lora:     $BASE_LORA_DIR"
echo "instruct_lora: $INSTRUCT_LORA_DIR"
echo "Port:          $PORT"
echo "Host:          $(hostname)"
echo "GPUs:          $(nvidia-smi -L)"
echo "==="

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

UV_EXCLUDE_NEWER="9998-12-31" uv run vllm serve "$BACKBONE" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 2 \
    --lora-modules "base_lora=$BASE_LORA_DIR" "instruct_lora=$INSTRUCT_LORA_DIR" \
    --trust-remote-code
