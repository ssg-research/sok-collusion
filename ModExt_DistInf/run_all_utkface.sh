#!/usr/bin/env bash
# UTKFace, both ratio pairs. Reproduces both UTKFACE columns of Table 6
# (paper §5.3) for all three Settings across 5 seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Stream progress (dataset extraction, per-epoch logs) live through tee/screen.
export PYTHONUNBUFFERED=1

for exp_id in 0 1 2 3 4; do
    for setting in 1 2 3; do
        echo "=== Setting ${setting} | ratio 0.45/0.55 ==="
        uv run python modext_distinf.py \
            --setting "${setting}" \
            --exp_id "${exp_id}" \
            --dataset utkface \
            --target_attribute gender \
            --filter_column race \
            --filter_value 0 \
            --train_subsample 1000 \
            --test_subsample 1000 \
            --ratio1 0.45 \
            --ratio2 0.55 \
            --output results/collusion_results_utkface.csv

        echo "=== Setting ${setting} | ratio 0.475/0.525 ==="
        uv run python modext_distinf.py \
            --setting "${setting}" \
            --exp_id "${exp_id}" \
            --dataset utkface \
            --target_attribute gender \
            --filter_column race \
            --filter_value 0 \
            --train_subsample 1000 \
            --test_subsample 1000 \
            --ratio1 0.475 \
            --ratio2 0.525 \
            --output results/collusion_results_utkface.csv
    done
done
