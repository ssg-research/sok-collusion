#!/usr/bin/env bash
# One-shot environment + data setup for the USENIX Security 2026 SoK
# colluding-adversaries artifact. Idempotent: safe to re-run.
#
#   bash setup.sh    Sets up everything for all three parts:
#                      - `uv sync` (amuletml + the pinned torch) and an import check.
#                      - Part C face data for AIA/DIA: builds the CelebA 64x64 cache
#                        (prep_celeba.py) and downloads the UTKFace CSV.
#                    CIFAR (Parts A/B and Part C's MIA) auto-downloads when those
#                    experiments first run, so nothing to do here. The reproduction
#                    reconstructs with the GAN-free geiping method, so GIFD/StyleGAN2
#                    is NOT needed (only the optional --method gifd needs it; README C.2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# GDrive share URL (or bare file ID) for Part C's UTKFace pixel CSV. Part C reads a
# grayscale 48x48 pixel CSV, not Amulet's color-JPG UTKFace loader (see README C.2),
# so it is provided separately. Override at the command line if needed:
# `UTKFACE_CSV_URL=... bash setup.sh`.
UTKFACE_CSV_URL="${UTKFACE_CSV_URL:-https://drive.google.com/file/d/1iLCJEu2bwVdd0SiZFzNMkzvhH3TjTWN4/view}"

for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            # Print the leading header comment block only (skip shebang, stop at
            # the first non-comment line).
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown option: ${arg} (try --help)" >&2; exit 2 ;;
    esac
done

# 1. Python environment: amuletml==0.5.1 + the pinned torch, from the single
#    pyproject.toml shared by all three parts.
echo "==> uv sync"
uv sync

# 2. Sanity check: amulet imports and its dataset loaders resolve.
echo "==> verifying amulet import"
uv run python -c "import amulet; from amulet.datasets import load_celeba, load_utkface; print('amulet ok')"

# 3. Part C face data for AIA/DIA (CelebA cache + UTKFace CSV). No GIFD/StyleGAN2:
#    the reproduction reconstructs with --method geiping.
PARTC="${SCRIPT_DIR}/DtRecon_MemAttDistInf"
CELEBA_CACHE="${PARTC}/data/celeba/celeba_64.pt"
UTKFACE_CSV="${PARTC}/data/utkface/utkface.csv"

echo "==> Part C: CelebA 64x64 cache (prep_celeba.py)"
if [ -f "${CELEBA_CACHE}" ]; then
    echo "    CelebA cache already present"
else
    ( cd "${PARTC}" && uv run python prep_celeba.py )
fi

echo "==> Part C: UTKFace CSV"
if [ -f "${UTKFACE_CSV}" ]; then
    echo "    UTKFace CSV already present"
elif [ -n "${UTKFACE_CSV_URL}" ]; then
    mkdir -p "$(dirname "${UTKFACE_CSV}")"
    uv run python - "${UTKFACE_CSV_URL}" "${UTKFACE_CSV}" <<'PY'
import sys
import gdown
src, out = sys.argv[1], sys.argv[2]
if src.startswith("http"):
    gdown.download(url=src, output=out, fuzzy=True, quiet=False)
else:
    gdown.download(id=src, output=out, quiet=False)
PY
else
    echo "    [TODO] UTKFACE_CSV_URL is unset; set it (a GDrive link) and re-run, or"
    echo "           place the file by hand at: ${UTKFACE_CSV}"
fi

echo
if [ -f "${CELEBA_CACHE}" ] && [ -f "${UTKFACE_CSV}" ]; then
    echo "Setup complete. All three parts are ready to run."
else
    echo "Setup complete for Parts A, B, and Part C's MIA. UTKFace CSV still missing"
    echo "(see [TODO] above) — needed only for Part C's AIA/DIA on UTKFace."
fi
echo "Reproduce:  bash Pois_ModExt/run_all.sh  |  ModExt_DistInf/run_all_*.sh  |  DtRecon_MemAttDistInf/run_all.sh"
