#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache="$here/.runtime"
mkdir -p "$cache/debs" "$cache/bookworm" "$cache/lib"
cd "$cache/debs"
if ! compgen -G 'libscalapack-openmpi2.2_*.deb' > /dev/null; then apt-get download libscalapack-openmpi2.2; fi
for file in libscalapack-openmpi2.2_*.deb; do dpkg-deb -x "$file" "$cache/lib"; done
cd "$cache/bookworm"
if [[ ! -f bookworm.deb ]]; then python3 "$here/fetch_binary.py"; fi
echo "368d8a8caafa07d184948f2eb8ba9b4becc324c2d3f7df64557ab81df384a8d5  bookworm.deb" | sha256sum -c -
dpkg-deb --fsys-tarfile bookworm.deb | tar -x ./usr/bin/pw.x
export LD_LIBRARY_PATH="$cache/lib/usr/lib/x86_64-linux-gnu"
ldd usr/bin/pw.x > "$here/bookworm-libraries.txt"
if grep -q 'not found' "$here/bookworm-libraries.txt"; then cat "$here/bookworm-libraries.txt"; exit 1; fi
cd "$here/pseudos"
echo "7e1db62c3e74c3cf88d7994fe857d2b156ca4d0bc00e780603642213356edceb  Ga.pbe-dn-kjpaw_psl.1.0.0.UPF" | sha256sum -c -
echo "c6420b82107b1fe96a798a0232093dacb0e0917dbb49bccbff5f845525b2810a  Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" | sha256sum -c -
