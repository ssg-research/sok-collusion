#!/usr/bin/env bash
# Smoke test for Pois -> ModExt: exercises every pipeline branch on CIFAR10
# at a single (poison_rate, query_size) tuple and a baseline, with --epochs 5
# instead of the default 200. Verifies the pipeline executes end-to-end and
# writes rows to pois_modext_smoke.csv. NOT a reproduction of paper Table
# tab:trteeval -- the numbers will be much noisier and lower than the full
# 200-epoch runs.
#
# Cells exercised:
#   1. Clean target training (poisoned_portion=0)  + extraction at 2500 queries
#   2. Poisoned target training (poisoned_portion=0.1) + extraction at 2500 queries
#
# This covers: BadNets poisoning, KnockoffNets-style extraction (the
# amulet.unauth_model_ownership.ModelExtraction attack), evaluation on
# clean and poisoned test sets, and the resume-from-cache check on a second
# run.
#
# Expected runtime on a single A100 80GB: roughly 10-15 minutes total
# (estimated, not measured). Run once and report back so we can populate
# the README time budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SMOKE_OUTPUT="${SCRIPT_DIR}/pois_modext_smoke.csv"

# Start from a clean smoke CSV so a re-run does not accumulate stale rows.
rm -f "${SMOKE_OUTPUT}"

for poison in 0.0 0.1; do
    echo "=== smoke dataset=cifar10 exp_id=0 poison=${poison} query_size=0.1 epochs=5 ==="
    uv run python pois_modext.py \
        --dataset cifar10 \
        --poisoned_portion "${poison}" \
        --query_size 0.1 \
        --exp_id 0 \
        --epochs 5 \
        --output "${SMOKE_OUTPUT}"
done

echo
echo "Smoke results:"
cat "${SMOKE_OUTPUT}"
