#!/usr/bin/env bash
# Full reproduction of Table tab:trteeval (paper section sec:trteeval):
# Poison -> ModExt on CIFAR10 and CIFAR100.
#
# Outer loop: dataset (cifar10, cifar100).
# Middle loop: poison rate (0, 0.05, 0.10, 0.15, 0.20 fraction-of-training).
#   These match the paper table's "0%", "5%", "10%", "15%", "20%" rows.
#   (The paper LaTeX renders the headers as "0.05%", "0.1%", etc., but the
#    actual code/values used to produce the table are fractions 0.05, 0.10,
#    etc. -- i.e. 5%, 10%, 15%, 20% of the training set is poisoned.)
# Inner loop: query budget (10/25/50/100% of D_aux, which is 25k records,
#   so 2500/6250/12500/25000 actual queries).
# Outermost loop: exp_id in {0, 1, 2} -> 3 seeds, matching the paper's std.
#
# A single (dataset, poison_rate, exp_id) tuple trains one target model and
# is then re-used across all four query budgets via the on-disk model cache.
# An interrupted run can be resumed by re-invoking this script: completed
# cells skip retraining.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for dataset in cifar10 cifar100; do
    for exp_id in 0 1 2; do
        for poison in 0.0 0.05 0.1 0.15 0.2; do
            for query_size in 1 0.5 0.25 0.1; do
                echo "=== dataset=${dataset} exp_id=${exp_id} poison=${poison} query_size=${query_size} ==="
                uv run python pois_modext.py \
                    --dataset "${dataset}" \
                    --poisoned_portion "${poison}" \
                    --query_size "${query_size}" \
                    --exp_id "${exp_id}"
            done
        done
    done
done
