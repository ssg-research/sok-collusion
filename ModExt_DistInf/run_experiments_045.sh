#!/usr/bin/env bash
# CelebA, alpha_1 = 0.45, alpha_2 = 0.55. Reproduces the first CELEBA column
# of Table tab:modextDIA for all three Settings across 5 seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for exp_id in 0 1 2 3 4; do
    for setting in 1 2 3; do
        echo "=== ratio=0.45, setting=${setting}, exp_id=${exp_id} ==="
        uv run python run_dist_inference_collusion.py \
            --setting "${setting}" \
            --exp_id "${exp_id}" \
            --ratio1 0.45 \
            --ratio2 0.55 \
            --output results/collusion_results_045.csv
    done
done
