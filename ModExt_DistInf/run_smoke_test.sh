#!/usr/bin/env bash
# Smoke test: runs all three settings on UTKFace at alpha=0.45/0.55, exp_id=0,
# with reduced training budget. Verifies the pipeline executes end-to-end and
# writes rows to results/collusion_smoke.csv. NOT a reproduction of the paper
# table - numbers will be noisier and lower than the published values.
#
# UTKFace is used here (rather than CelebA) because it is much smaller and
# makes the smoke test finish in roughly 10 minutes on an A100 80GB instead
# of the multiple hours a CelebA smoke would take. The full-reproduction
# scripts cover CelebA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for setting in 1 2 3; do
    echo "=== smoke setting=${setting} ==="
    uv run python modext_distinf.py \
        --setting "${setting}" \
        --exp_id 0 \
        --dataset utkface \
        --target_attribute gender \
        --filter_column race \
        --filter_value 0 \
        --train_subsample 1000 \
        --test_subsample 1000 \
        --ratio1 0.45 \
        --ratio2 0.55 \
        --num_models 4 \
        --epochs 3 \
        --extraction_epochs 2 \
        --output results/collusion_smoke.csv
done

echo
echo "Smoke results:"
cat results/collusion_smoke.csv
