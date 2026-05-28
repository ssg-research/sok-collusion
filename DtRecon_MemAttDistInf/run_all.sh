#!/usr/bin/env bash
# Full reproduction of paper section 5.4 (DataRecon -> {MIA, AIA, DIA}), reported
# as prose; reference numbers in README C.5. Each attack is a self-contained block
# (reconstruct, then sweep) so blocks can run on separate GPUs. See the time
# budget in README C.2.
#
# Prerequisites (README C.2): `uv sync`; for AIA/DIA also the GIFD checkout
# ($GIFD_ROOT or ./GIFD) + the StyleGAN2-FFHQ checkpoint + prepared face data
# (CelebA 64x64 tensor via prep_celeba.py, UTKFace csv).
#
# Knobs you may want to flip:
#   MIA_SEEDS   "0"          -> single-seed run that fills the C.5 table.
#               "0 1 2 3 4"  -> 5-seed mean +/- std (target_seed fixed at 0).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OVERLAPS=(0.0 0.25 0.5 0.75)
GIFD_ROOT="${GIFD_ROOT:-${SCRIPT_DIR}/GIFD}"
MIA_SEEDS="${MIA_SEEDS:-0}"          # C.5 numbers are single-seed (seed 0)

# ── DataRecon -> MIA: CIFAR10/100, LiRA, ResNet-34 ──────────────────────────
# Pool: 100 init-round Geiping reconstructions of target members (no GIFD).
# pool_mode=budget reproduces the reported numbers: n_in = overlap * recon_budget
# distinct recons injected into a 15k-record shadow pool. num_shadow=64.
for ds in cifar10 cifar100; do
    pool="recon_pools/${ds}_seed0"
    echo "### MIA reconstruct: ${ds} (100 records) ###"
    uv run python cifar_recon.py --dataset "${ds}" --target_seed 0 \
        --n_records 100 --batch_size 64 --iters 2000 --output_dir "${pool}"
    for seed in ${MIA_SEEDS}; do
        for p in "${OVERLAPS[@]}"; do
            echo "### MIA: ${ds} overlap=${p} seed=${seed} ###"
            uv run python mia_overlap.py --dataset "${ds}" --overlap_p "${p}" \
                --seed "${seed}" --target_seed 0 --num_shadow 64 --n_a 15000 \
                --recon_source gifd --recon_dir "${pool}" --recon_budget 100 \
                --pool_mode budget --results_csv "results/mia_${ds}.csv"
        done
    done
done

# ── AIA / DIA need GIFD + face data ─────────────────────────────────────────
if [ ! -d "${GIFD_ROOT}" ] || [ ! -f "data/celeba/celeba_64.pt" ]; then
    echo "### AIA/DIA skipped: need GIFD at ${GIFD_ROOT} and prepared face data (README C.2)."
    exit 0
fi

# ── DataRecon -> AIA: CelebA/UTKFace, shadow attribute-inference probe, ResNet-18@64
# n=15 per cell: three target models (seed 0,1,2) x five adversary seeds (0..4).
# Headline config that produced the C.5 numbers: targets are OVERFIT on the first 600
# records of D_train (weight_decay=0, 120 epochs), tagged of600 / sex_of600, which
# sharpens the member/non-member gap the probe exploits. Per dataset: CelebA n_dout=300,
# recon_budget=200, N_AUG=3; UTKFace n_dout=150, recon_budget=100, N_AUG=5. The Geiping
# pools are 100 records and target-independent (built from an init-model gradient).
for ds in celeba utkface; do
    if [ "${ds}" = utkface ]; then
        attr="--y_attr sex --z_attr race"; ttask=sex_of600; ndout=150; budget=100; naug=5
    else
        attr=""; ttask=of600; ndout=300; budget=200; naug=3
    fi
    for tseed in 0 1 2; do
        echo "### AIA train overfit target: ${ds} seed=${tseed} (${ttask}) ###"
        uv run python face_targets.py --dataset "${ds}" --seed "${tseed}" \
            --task "${ttask}" --n_train 600 --weight_decay 0 --epochs 120 ${attr}
        pool="recon_pools/${ds}_seed${tseed}_geiping"
        echo "### AIA reconstruct: ${ds} target_seed=${tseed} (100 records, GIFD/Geiping) ###"
        uv run python face_recon.py --dataset "${ds}" --target_seed "${tseed}" \
            --method geiping --geiping_iters 6000 --restarts 1 --n_records 100 ${attr} --output_dir "${pool}"
        for seed in 0 1 2 3 4; do
            for p in "${OVERLAPS[@]}"; do
                echo "### AIA: ${ds} target_seed=${tseed} seed=${seed} overlap=${p} ###"
                uv run python aia_image.py --dataset "${ds}" --target_seed "${tseed}" \
                    --seed "${seed}" --overlap "${p}" --recon_source gifd --recon_dir "${pool}" \
                    --target_task "${ttask}" ${attr} --n_dout "${ndout}" --recon_budget "${budget}" \
                    --n_aug "${naug}" --results_csv "results/aia_${ds}.csv"
            done
        done
    done
done

# ── DataRecon -> DIA: CelebA/UTKFace, Suri et al. black-box, ResNet-18@64 ────
# alpha_1 (victim) = 0.5 vs alpha_2 in {0.1 (task lo), 0.9 (task hi)}; n=3 seeds,
# 64 shadows per ratio. Reuses the seed-0 AIA reconstruction pool; if the AIA block
# was not run (e.g. blocks split across GPUs) the pool is rebuilt here first.
for ds in celeba utkface; do
    [ "${ds}" = utkface ] && attr="--y_attr sex --z_attr race" || attr=""
    pool="recon_pools/${ds}_seed0_geiping"
    if [ ! -d "${pool}" ]; then
        echo "### DIA: ${ds} seed-0 pool missing; reconstructing (target-independent) ###"
        uv run python face_recon.py --dataset "${ds}" --target_seed 0 \
            --method geiping --geiping_iters 6000 --restarts 1 --n_records 100 ${attr} --output_dir "${pool}"
    fi
    for task in lo hi; do
        for seed in 0 1 2; do
            for p in "${OVERLAPS[@]}"; do
                echo "### DIA: ${ds} task=${task} seed=${seed} overlap=${p} ###"
                uv run python dia_image.py --dataset "${ds}" --task "${task}" \
                    --target_seed 0 --seed "${seed}" --overlap "${p}" \
                    --recon_source gifd --recon_dir "${pool}" \
                    --n_shadow 64 --results_csv "results/dia_${ds}.csv"
            done
        done
    done
done

echo "### run_all done. Render paper tables with: uv run python generate_teeval_tables.py ###"
