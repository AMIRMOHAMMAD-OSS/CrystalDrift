#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache="$here/.runtime"
mkdir -p "$cache/bookworm"
cd "$cache/bookworm"
url="https://deb.debian.org/debian/pool/main/e/espresso/quantum-espresso_6.7-2+b1_amd64.deb"
if [[ ! -f bookworm.deb ]]; then curl -fsSL --retry 2 --connect-timeout 20 "$url" -o bookworm.part; mv bookworm.part bookworm.deb; fi
echo "368d8a8caafa07d184948f2eb8ba9b4becc324c2d3f7df64557ab81df384a8d5  bookworm.deb" | sha256sum -c -
dpkg-deb --fsys-tarfile bookworm.deb | tar -x ./usr/bin/pw.x
export LD_LIBRARY_PATH="$cache/lib/usr/lib/x86_64-linux-gnu"
ldd usr/bin/pw.x | tee "$here/bookworm-libraries.txt"
if grep -q 'not found' "$here/bookworm-libraries.txt"; then exit 1; fi
export QE_PW="$PWD/usr/bin/pw.x"
bash "$here/run.sh"
