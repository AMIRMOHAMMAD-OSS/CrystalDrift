#!/usr/bin/env bash
set -eu
base=lattice_symmetry/v3/models
pid="$1"
while [ ! -s "$base/fits/small-angular-laplace-23/complete.json" ] || [ ! -s "$base/lookup-cache-lifetime.json" ]; do sleep 1; done
commandline=$(tr '\000' ' ' < "/proc/$pid/cmdline")
case "$commandline" in
    *"python -u -m lattice_symmetry.v3.models.window --coordinated "*) ;;
    *) exit 2 ;;
esac
[ "$(readlink -f "/proc/$pid/cwd")" = "$(pwd -P)" ]
kill -TERM "$pid"
while [ -d "/proc/$pid" ]; do sleep 1; done
exec bash "$base/resume_local.sh"

