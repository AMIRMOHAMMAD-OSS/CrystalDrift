#!/usr/bin/env bash
# Run after the main capacity program releases CUDA. Each rerun resumes its own optimizer.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CRYTALDRIFT_NUMBA_SITE=/home/amir/miniconda3/envs/scrna/lib/python3.11/site-packages
export NUMBA_CACHE_DIR=/tmp/crytaldrift-v3-20261121/jit
export CRYTALDRIFT_SCRATCH=/tmp/crytaldrift-v3-20261121/checkpoints
folder=lattice_symmetry/v3/models
while [ ! -s "$folder/window-complete.json" ]; do sleep 5; done
test -s "$folder/system-auxiliary-input-test.json"
/home/amir/miniconda3/envs/mp_lstm/bin/python -u "$folder/gpu.py" --tier small --arm angular --system-auxiliary --seed 11 --batch 16 --device cuda
