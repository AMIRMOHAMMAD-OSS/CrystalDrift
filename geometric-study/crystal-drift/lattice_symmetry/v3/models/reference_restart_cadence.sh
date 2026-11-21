#!/usr/bin/env bash
# Adopt the lower-I/O restart interval after the second fully committed reference optimizer.
set -euo pipefail
base=lattice_symmetry/v3/models
optimizer=/tmp/crytaldrift-v3-20261121/checkpoints/reference-contacts-laplace-11/resume.pt
pid=$(pgrep -f '^/home/amir/miniconda3/envs/mp_lstm/bin/python -u -m lattice_symmetry.v3.models.window --coordinated$')
test "$(readlink -f "/proc/$pid/cwd")" = "$(pwd -P)"
first=$(date -d '2026-11-21 09:13:41 +0330' +%s)
while [ "$(stat -c %Y "$optimizer")" -le "$first" ]; do sleep 1; done
kill -TERM "$pid"
while [ -d "/proc/$pid" ]; do sleep 1; done
date -u '+{"utc":"%FT%TZ","early_shards":[12,24],"subsequent_periodic_shards":48,"milestones":["half","full"],"reason":"Measured first 1.63 GB atomic checkpoint incurred about 190 s of write overhead."}' > "$base/reference-restart-cadence.json"
exec bash "$base/resume_local.sh"
