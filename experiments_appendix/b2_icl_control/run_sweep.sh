#!/bin/bash
# §B.2 ICL K-sweep for a single seed. Runs K ∈ {1, 2, 3, 4, 5, 8, 12, 16, 20, 50}
# (override with K_VALUES env var).
#
# Results land under experiments_appendix/b2_icl_control/results/seed{SEED}/ so
# seed runs do not overwrite each other (the upstream runner's label is
# seed-agnostic).
#
# K=0 (no ICL prefix) has no seed dependence — run it once via the unmodified
# eval_config.yaml with icl_n: 0, not from this script.
#
# Run from repo root:
#   SEED=42 bash experiments_appendix/b2_icl_control/run_sweep.sh
#   SEED=43 bash experiments_appendix/b2_icl_control/run_sweep.sh
#   ...
# or use run_all_seeds.sh to iterate over all five seeds.

set -euo pipefail

TEMPLATE="experiments_appendix/b2_icl_control/eval_config.yaml"
SEED="${SEED:-42}"
OUTPUT_DIR="experiments_appendix/b2_icl_control/results/seed${SEED}"
K_VALUES="${K_VALUES:-1 2 3 4 5 8 12 16 20 50}"

LOG_DIR="experiments_appendix/b2_icl_control/.logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

for K in $K_VALUES; do
    echo ""
    echo "=============================================="
    echo " ICL sweep — seed=${SEED}  K=${K}"
    echo "=============================================="

    TMP_CFG=$(mktemp -t "icl_s${SEED}_k${K}_XXXX.yaml")
    sed -e "s/^icl_n:.*/icl_n: ${K}/" \
        -e "s/^icl_seed:.*/icl_seed: ${SEED}/" \
        -e "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" \
        "$TEMPLATE" > "$TMP_CFG"

    LOG="${LOG_DIR}/seed${SEED}_k${K}.log"
    echo "  config: $TMP_CFG"
    echo "  output: $OUTPUT_DIR"
    echo "  log:    $LOG"

    uv run python -m src.evals sweep "$TMP_CFG" 2>&1 | tee "$LOG"

    rm -f "$TMP_CFG"
done

echo ""
echo "Seed ${SEED} complete. Results in ${OUTPUT_DIR}/."
