#!/usr/bin/env bash
# Minimal full sweep: every (attack, dataset, overlap) at a reduced budget (few
# epochs / shadows / reconstructions, single seed). Hints at the trend; does NOT
# reproduce the paper numbers (use run_all.sh for that). Target ~30-60 min on one
# GPU, dominated by the face reconstructions.
#
# MIA needs only CIFAR (auto-downloaded) and runs unconditionally. AIA and DIA
# need the GIFD checkout + prepared face data (README C.2); if either is missing
# this script runs MIA and skips AIA/DIA with a notice rather than failing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OVERLAPS=(0.0 0.25 0.5 0.75)
GIFD_ROOT="${GIFD_ROOT:-${SCRIPT_DIR}/GIFD}"

# ── DataRecon -> MIA (CIFAR10, CIFAR100) ────────────────────────────────────
for ds in cifar10 cifar100; do
    pool="recon_pools/min_${ds}"
    echo "=== MIA reconstruct: ${ds} (64 records, 500 iters) ==="
    uv run python cifar_recon.py --dataset "${ds}" --target_seed 0 \
        --n_records 64 --batch_size 64 --iters 500 --output_dir "${pool}"
    for p in "${OVERLAPS[@]}"; do
        echo "=== MIA: ${ds} overlap=${p} (8 shadows, 10 epochs) ==="
        uv run python mia_overlap.py --dataset "${ds}" --overlap_p "${p}" \
            --seed 0 --target_seed 0 --num_shadow 8 --epochs 10 --n_a 2000 \
            --recon_source gifd --recon_dir "${pool}" --recon_budget 64 --pool_mode budget \
            --results_csv "results/min_mia_${ds}.csv"
    done
done

# ── DataRecon -> AIA / DIA (CelebA, UTKFace) ────────────────────────────────
if [ ! -d "${GIFD_ROOT}" ] || [ ! -f "data/celeba/celeba_64.pt" ]; then
    echo "### AIA/DIA skipped: need GIFD at ${GIFD_ROOT} and prepared face data."
    echo "### See README C.2 for the one-time GIFD + face-data setup, then re-run."
    exit 0
fi

for ds in celeba utkface; do
    if [ "${ds}" = utkface ]; then attr="--y_attr sex --z_attr race"; ttask=sex_of600; else attr=""; ttask=of600; fi
    pool="recon_pools/min_${ds}_faces"
    echo "=== train overfit target: ${ds} seed 0 (${ttask}, reduced epochs) ==="
    uv run python face_targets.py --dataset "${ds}" --seed 0 \
        --task "${ttask}" --n_train 600 --weight_decay 0 --epochs 40 ${attr}
    echo "=== face reconstruct: ${ds} (32 records, reduced iters) ==="
    uv run python face_recon.py --dataset "${ds}" --target_seed 0 --method geiping \
        --geiping_iters 800 --restarts 1 --n_records 32 ${attr} --output_dir "${pool}"
    for p in "${OVERLAPS[@]}"; do
        echo "=== AIA: ${ds} overlap=${p} ==="
        uv run python aia_image.py --dataset "${ds}" --target_seed 0 --seed 0 \
            --overlap "${p}" --recon_source gifd --recon_dir "${pool}" \
            --target_task "${ttask}" ${attr} \
            --n_dout 100 --recon_budget 32 --results_csv "results/min_aia_${ds}.csv"
    done
    echo "=== DIA: ${ds} task=lo (8 shadows) ==="
    for p in "${OVERLAPS[@]}"; do
        uv run python dia_image.py --dataset "${ds}" --task lo --target_seed 0 --seed 0 \
            --overlap "${p}" --recon_source gifd --recon_dir "${pool}" \
            --n_shadow 8 --recon_budget 32 --results_csv "results/min_dia_${ds}.csv"
    done
done
echo "=== run_minimal done; CSVs in results/min_*.csv ==="
