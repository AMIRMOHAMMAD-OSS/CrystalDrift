#!/usr/bin/env bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRYTALDRIFT_NUMBA_SITE="${CRYTALDRIFT_NUMBA_SITE:-/home/amir/miniconda3/envs/scrna/lib/python3.11/site-packages}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/crytaldrift-v3-numba}"
PYTHON="${CRYTALDRIFT_PYTHON:-/home/amir/miniconda3/envs/mp_lstm/bin/python}"
DIR=lattice_symmetry/v3/models
run () {
    "$PYTHON" -u "$DIR/gpu.py" "$@" 2>&1 | tee -a "$DIR/gpu.log"
    return "${PIPESTATUS[0]}"
}
run --tier small --arm angular
run --tier medium --arm angular
run --tier small --arm contacts
run --tier small --arm reciprocal
run --tier small --arm meanflow
run --tier medium --arm meanflow
run --tier small --arm contacts --affinity gaussian
run --tier small --arm angular --affinity gaussian
run --tier small --arm pair
run --tier small --arm center0
run --tier small --arm variance
run --tier tiny --arm angular
run --tier small --arm angular --local

