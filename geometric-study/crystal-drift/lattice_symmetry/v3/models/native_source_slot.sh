#!/usr/bin/env bash
# Second, short handoff for exact released-native Gaussian decoding and the sparse full-model replay.
set -euo pipefail
base=lattice_symmetry/v3/models
case "${1:-radial}" in
  radial) finished="$base/fits/small-radial4-laplace-11/complete.json"; slot=uflow-source ;;
  large) finished="$base/fits/large-angular-laplace-11/complete.json"; slot=uflow-source-final ;;
  *) exit 2 ;;
esac
pid=$(pgrep -f '^/home/amir/miniconda3/envs/mp_lstm/bin/python -u -m lattice_symmetry.v3.models.window --coordinated$')
test "$(readlink -f "/proc/$pid/cwd")" = "$(pwd -P)"
while [ ! -s "$finished" ]; do sleep 1; done
kill -TERM "$pid"
while [ -d "/proc/$pid" ]; do sleep 1; done
date -u '+{"released_utc":"%FT%TZ","handoff":"exact released native zero prior verification after full optimizer checkpoint"}' > "$base/$slot-exclusive.json"
while [ ! -s "$base/$slot-finished.json" ]; do sleep 3; done
exec bash "$base/resume_local.sh"
