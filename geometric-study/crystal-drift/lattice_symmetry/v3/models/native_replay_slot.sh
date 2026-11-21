#!/usr/bin/env bash
# Requested late exclusive GPU interval for paired native reduced-model replay.
set -euo pipefail
base=lattice_symmetry/v3/models
optimizer=/tmp/crytaldrift-v3-20261121/checkpoints/small-radial4-laplace-11/resume.pt
pid=$(pgrep -f '^/home/amir/miniconda3/envs/mp_lstm/bin/python -u -m lattice_symmetry.v3.models.window --coordinated$')
test "$(readlink -f "/proc/$pid/cwd")" = "$(pwd -P)"
previous=$(stat -c %Y "$optimizer")
while [ "$(stat -c %Y "$optimizer")" = "$previous" ]; do sleep 1; done
kill -TERM "$pid"
while [ -d "/proc/$pid" ]; do sleep 1; done
date -u '+{"released_utc":"%FT%TZ","handoff":"native replay after atomic radial-control checkpoint"}' > "$base/uflow-replication-exclusive.json"
while [ ! -s "$base/uflow-replication-finished.json" ]; do sleep 5; done
exec bash "$base/resume_local.sh"
