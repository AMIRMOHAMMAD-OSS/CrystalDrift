#!/usr/bin/env bash
# This one-time rollout preserves the completed matched-order medium fit before replacing its supervisor.
set -eu
pid="$1"
base=lattice_symmetry/v3/models
while [ ! -s "$base/fits/medium-meanflow-laplace-11-commonorder/complete.json" ]; do sleep 1; done
commandline=$(tr '\000' ' ' < "/proc/$pid/cmdline")
case "$commandline" in
    *"python -u -m lattice_symmetry.v3.models.window --coordinated "*) ;;
    *) exit 2 ;;
esac
[ "$(readlink -f "/proc/$pid/cwd")" = "$(pwd -P)" ]
kill -TERM "$pid"
while [ -d "/proc/$pid" ]; do sleep 1; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/crytaldrift-v3-20261121/jit
export MATCHER_PYTHON=/tmp/crytaldrift-v3-20261121/matcher-env/bin/python
export PYTHONPATH="$(pwd -P)"
export CRYTALDRIFT_NUMBA_SITE=/home/amir/miniconda3/envs/scrna/lib/python3.11/site-packages
py=/home/amir/miniconda3/envs/mp_lstm/bin/python
"$py" -u -m lattice_symmetry.v3.models.development_device --device cuda --native-cache
exec bash "$base/resume_local.sh"

