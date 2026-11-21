#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache="${QE_CACHE:-$here/.runtime}"
mkdir -p "$cache"
binary="${QE_PW:-$cache/bookworm/usr/bin/pw.x}"
if [[ ! -x "$binary" ]]; then bash "$here/setup.sh"; fi
export LD_LIBRARY_PATH="$cache/lib/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$here"
# Keep scratch files local to each independent electronic calculation.
short=$(mktemp -d /tmp/cdqe.XXXXXX)
ln -s "$binary" "$short/pw.x"
ln -s "$here/pseudos" "$short/pp"
ulimit -c 0
exec 9>"$cache/experiments.lock"
flock 9
run_one() {
    local template="$1" job work
    job="$(basename "$template" .in)"
    work="$short/$job"; mkdir -p "$work"
    if [[ -s "$job.out" ]] && grep -q 'JOB DONE' "$job.out" && grep -q 'convergence has been achieved' "$job.out"; then return 0; fi
    sed -e "s|PSEUDODIR|../pp|" -e "s|SCRATCHDIR|./|" "$template" > "$work/pw.in"
    (cd "$work" && ../pw.x -in pw.in) > "$job.running" 2>&1 || { mv "$job.running" "$job.out"; return 1; }
    if ! grep -q 'JOB DONE' "$job.running" || ! grep -q 'convergence has been achieved' "$job.running"; then
        mv "$job.running" "$job.out"
        return 1
    fi
    mv "$job.running" "$job.out"
    echo "Completed $job"
}
export short
export -f run_one
for mesh in 2 3; do
    printf '%s\0' *-k"$mesh".in | xargs -0 -P 4 -I '{}' bash -c 'run_one "$1"' _ '{}'
done
