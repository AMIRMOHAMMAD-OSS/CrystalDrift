#!/usr/bin/env bash
#SBATCH --job-name=crytaldrift-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
set -euo pipefail
# Launch from the release root, with the authorized allocation and installed interpreters supplied by the caller.
: "${CRYTALDRIFT_PYTHON:?Set to the CUDA-enabled training interpreter}"
: "${MATCHER_PYTHON:?Set to the pymatgen 2024.4.13 matcher interpreter}"
tier="${CRYTALDRIFT_TIER:-reference}"
microbatch="${CRYTALDRIFT_BATCH:-2}"
arm="${CRYTALDRIFT_ARM:-angular}"
seed="${CRYTALDRIFT_SEED:-11}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
if [ ! -e lattice_symmetry/v3/models/cache/calibration.npz ]; then
    "$CRYTALDRIFT_PYTHON" -m lattice_symmetry.v3.models.prepare
fi
"$CRYTALDRIFT_PYTHON" -u -m lattice_symmetry.v3.models.train --tier "$tier" --arm "$arm" --batch "$microbatch" --seed "$seed" --device cuda
"$CRYTALDRIFT_PYTHON" -u -m lattice_symmetry.v3.models.select_weights --confirm

