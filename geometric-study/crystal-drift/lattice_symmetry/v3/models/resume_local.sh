#!/usr/bin/env bash
# Real local re-launch used after the primary seed-11 comparator rotation.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/crytaldrift-v3-20261121/jit
export MATCHER_PYTHON=/tmp/crytaldrift-v3-20261121/matcher-env/bin/python
export CRYTALDRIFT_NUMBA_SITE=/home/amir/miniconda3/envs/scrna/lib/python3.11/site-packages
export CRYTALDRIFT_BANK_SCRATCH=/tmp/crytaldrift-v3-20261121/banks
export CRYTALDRIFT_SCRATCH=/tmp/crytaldrift-v3-20261121/checkpoints
export CRYTALDRIFT_WINDOWS_CONFIRM=1
dir=lattice_symmetry/v3/models
python=/home/amir/miniconda3/envs/mp_lstm/bin/python
while [ ! -e "$dir/uflow-finished.json" ]; do sleep 5; done
while true; do
    code=0
    "$python" -u -m lattice_symmetry.v3.models.window --coordinated || code=$?
    if [ "$code" -eq 0 ]; then break; fi
    if [ "$code" -ne 75 ]; then exit "$code"; fi
    while [ ! -e "$dir/timing-finished.json" ]; do sleep 5; done
done

