#!/bin/bash
#SBATCH --job-name=crytaldrift
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=18:00:00
#SBATCH --output=crytaldrift-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the reconstructed crystal-drift root}"
module load arch/a100
: "${CRYTALDRIFT_PYTHON:?Set CRYTALDRIFT_PYTHON to the prepared absolute CUDA Python executable}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CRYTALDRIFT_SCRATCH="${CRYTALDRIFT_SCRATCH:-${SCRATCH:?}/crytaldrift/checkpoints}"
export CRYTALDRIFT_BANK_SCRATCH="${CRYTALDRIFT_BANK_SCRATCH:-${SCRATCH:?}/crytaldrift/banks}"
export NUMBA_CACHE_DIR="${SCRATCH:?}/crytaldrift/numba"
mkdir -p "$CRYTALDRIFT_SCRATCH" "$CRYTALDRIFT_BANK_SCRATCH" "$NUMBA_CACHE_DIR"
srun "$CRYTALDRIFT_PYTHON" -c 'import torch, numba; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())'
nvidia-smi
srun "$CRYTALDRIFT_PYTHON" -m lattice_symmetry.v3.models.train "$@"

