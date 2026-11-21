#!/usr/bin/env bash
# Enable a private single-threaded BLAS for subsequent meshes. Running jobs keep
# the library already mapped at startup. The electronic Hamiltonian is unchanged.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache="$here/.runtime"
cd "$cache/debs"
pkg=libopenblas0-pthread
if ! compgen -G "$pkg"'_*.deb' > /dev/null; then apt-get download "$pkg"; fi
for item in "$pkg"_*.deb; do dpkg-deb -x "$item" "$cache/lib"; done
private="$cache/lib/usr/lib/x86_64-linux-gnu"
for name in libblas.so.3 liblapack.so.3 libopenblas.so.0; do
    test -s "$private/openblas-pthread/$name"
    if [[ ! -e "$private/$name" ]]; then ln -s "openblas-pthread/$name" "$private/$name"; fi
done
export LD_LIBRARY_PATH="$private/openblas-pthread:$private"
ldd "$cache/bookworm/usr/bin/pw.x" > "$here/optimized-libraries.txt"
dpkg-deb -f "$item" Package Version
