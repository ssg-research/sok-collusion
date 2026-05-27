#!/usr/bin/env bash
# Minimal-budget full configuration sweep for Pois -> ModExt.
#
# Purpose: produce a realistic per-cell timing distribution to project the
# 200-epoch reproduction (run_all.sh). NOT a paper reproduction -- numbers
# at --epochs 10 will be far below the published table.
#
# Sweep: 2 datasets x 2 poison rates x 4 query sizes x exp_id=0 = 16 cells.
# Poisons 0.0 and 0.1 exercise both the clean and BadNets training branches.
# 4 unique target models are trained (2 datasets x 2 poisons); each is
# re-used across the four query budgets via the on-disk model cache.
#
# Output:
#   results/pois_modext_minimal.csv     all 16 rows, each carrying a timestamp
#   logs/run_minimal_full_<UTC>.log     tee'd stdout/stderr for screen sessions
#
# Expected runtime on a single A100 80GB: roughly 20-30 minutes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/results" "${SCRIPT_DIR}/logs"
OUTPUT="${SCRIPT_DIR}/results/pois_modext_minimal.csv"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${SCRIPT_DIR}/logs/run_minimal_full_${TS}.log"

# Start from a clean output CSV so deltas in the timestamp column are
# unambiguous: the first row's wall time is unknown, every subsequent
# row's delta is the wall time of the cell that produced it.
rm -f "${OUTPUT}"

{
    echo "=== run_minimal_full started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    for dataset in cifar10 cifar100; do
        for poison in 0.0 0.1; do
            for query_size in 0.1 0.25 0.5 1; do
                echo
                echo "=== dataset=${dataset} poison=${poison} query=${query_size} epochs=10 exp_id=0 ==="
                uv run python pois_modext.py \
                    --dataset "${dataset}" \
                    --poisoned_portion "${poison}" \
                    --query_size "${query_size}" \
                    --exp_id 0 \
                    --epochs 10 \
                    --output "${OUTPUT}"
            done
        done
    done
    echo
    echo "=== run_minimal_full finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo
    echo "Per-cell timings can be derived from the timestamp column of:"
    echo "  ${OUTPUT}"
    echo
    echo "Final CSV:"
    cat "${OUTPUT}"
} 2>&1 | tee "${LOG}"
