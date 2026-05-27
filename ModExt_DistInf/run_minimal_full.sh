#!/usr/bin/env bash
# Minimal-budget full configuration sweep for ModExt -> DistInf.
#
# Purpose: produce a realistic per-cell timing distribution to project the
# full 5-seed reproduction (run_experiments_045/_0475/_utkface.sh). NOT a
# paper reproduction -- numbers at --num_models 4 / --epochs 3 will be far
# below the published table.
#
# Sweep: 2 datasets x 2 ratio pairs x 3 settings x exp_id=0 = 12 cells.
# Reduced per-cell budget matches the existing smoke test (smoke covers 3
# cells; this script covers all 12 to expose both per-dataset and
# per-setting timing variance).
#
# Output:
#   results/collusion_results_minimal.csv  all 12 rows, each carrying a timestamp
#   logs/run_minimal_full_<UTC>.log        tee'd stdout/stderr for screen sessions
#
# Expected runtime on a single A100 80GB: roughly 30-60 minutes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/results" "${SCRIPT_DIR}/logs"
OUTPUT="${SCRIPT_DIR}/results/collusion_results_minimal.csv"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${SCRIPT_DIR}/logs/run_minimal_full_${TS}.log"

rm -f "${OUTPUT}"

# Shared smoke-budget args (deliberately identical to run_smoke_test.sh so
# per-cell times can be compared against the smoke numbers).
COMMON_BUDGET=(
    --num_models 4
    --epochs 3
    --extraction_epochs 2
    --train_subsample 1000
    --test_subsample 1000
    --exp_id 0
    --output "${OUTPUT}"
)

run_celeba_cell() {
    local setting="$1" r1="$2" r2="$3"
    echo
    echo "=== dataset=celeba setting=${setting} ratio=${r1}/${r2} ==="
    uv run python modext_distinf.py \
        --setting "${setting}" \
        --dataset celeba \
        --ratio1 "${r1}" \
        --ratio2 "${r2}" \
        "${COMMON_BUDGET[@]}"
}

run_utkface_cell() {
    local setting="$1" r1="$2" r2="$3"
    echo
    echo "=== dataset=utkface setting=${setting} ratio=${r1}/${r2} ==="
    uv run python modext_distinf.py \
        --setting "${setting}" \
        --dataset utkface \
        --target_attribute gender \
        --filter_column race \
        --filter_value 0 \
        --ratio1 "${r1}" \
        --ratio2 "${r2}" \
        "${COMMON_BUDGET[@]}"
}

{
    echo "=== run_minimal_full started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    for setting in 1 2 3; do
        for ratios in "0.45 0.55" "0.475 0.525"; do
            # shellcheck disable=SC2086
            set -- ${ratios}
            run_celeba_cell "${setting}" "$1" "$2"
            run_utkface_cell "${setting}" "$1" "$2"
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
