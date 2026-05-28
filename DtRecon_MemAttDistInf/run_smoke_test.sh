#!/usr/bin/env bash
# Smoke test: one DataRecon -> MIA cell end-to-end on CIFAR10, single epoch-scale
# budget. Confirms the environment works (amulet import, GIFD-free CIFAR
# reconstruction, LiRA shadow training, metric computation). Does NOT reproduce
# any paper number. Target ~3-5 min on one GPU; CPU-only works but is slower.
#
# Exercises the cheapest path (MIA needs no GIFD, no face data). If this passes,
# the AIA/DIA paths additionally need the GIFD checkout + face data (see README C.2).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

POOL=recon_pools/smoke_cifar10

echo "=== [1/2] reconstruct 16 CIFAR10 records (tiny, 200 iters) ==="
uv run python cifar_recon.py \
    --dataset cifar10 --target_seed 0 --n_records 16 \
    --batch_size 16 --iters 200 --output_dir "${POOL}"

echo "=== [2/2] one MIA cell: cifar10, overlap=0.5, 2 shadows, 2 epochs ==="
uv run python mia_overlap.py \
    --dataset cifar10 --overlap_p 0.5 --seed 0 --target_seed 0 \
    --num_shadow 2 --epochs 2 --n_a 500 \
    --recon_source gifd --recon_dir "${POOL}" --recon_budget 16 --pool_mode budget \
    --results_csv results/smoke_mia.csv

echo "=== smoke test OK: see results/smoke_mia.csv ==="
