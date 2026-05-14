#!/bin/bash
# §B.2 ICL K-sweep across 5 seeds {42, 43, 44, 45, 46}, sequential. Each
# seed produces a full K-sweep {1, 2, 3, 4, 5, 8, 12, 16, 20, 50} under
# experiments_appendix/b2_icl_control/results/seed{S}/.
#
# Run from repo root:
#   bash experiments_appendix/b2_icl_control/run_all_seeds.sh

set -euo pipefail

SEEDS="${SEEDS:-42 43 44 45 46}"

for S in $SEEDS; do
    echo ""
    echo "################################################################"
    echo "# SEED ${S}"
    echo "################################################################"
    SEED="$S" bash experiments_appendix/b2_icl_control/run_sweep.sh
done

echo ""
echo "All seeds complete. Results in experiments_appendix/b2_icl_control/results/."
