#!/bin/bash
# Judge-model sweep — three-stage pipeline.
#
# Run from repo root.
# Re-running is safe: sample.parquet is deterministic (seed 42); judge.py
# caches every call to disk.

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== 1/3  Build stratified sample ==="
uv run python experiments_appendix/c8_judge_sweep/sample.py

echo
echo "=== 2/3  Call all 5 judges ==="
uv run python experiments_appendix/c8_judge_sweep/judge.py

echo
echo "=== 3/3  Compute reliability metrics + figures ==="
uv run python experiments_appendix/c8_judge_sweep/analyse.py
