"""Fetch the distribution-signed Debian Bookworm pw.x package and verify SHA-256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import os
import time
import urllib.request

size = 47638100
url = "https://deb.debian.org/debian/pool/main/e/espresso/quantum-espresso_6.7-2+b1_amd64.deb"
sha = "368d8a8caafa07d184948f2eb8ba9b4becc324c2d3f7df64557ab81df384a8d5"
here = Path(__file__).resolve().parent/".runtime/bookworm"
here.mkdir(parents=True, exist_ok=True)
chunk = 2**20

def get(i):
    lo, hi = i*chunk,min((i+1)*chunk,size)-1
    path = here/f"package-{lo}.part"
    if path.exists() and path.stat().st_size == hi-lo+1:return path
    for attempt in range(6):
        req=urllib.request.Request(url+f"?range={lo}&try={attempt+1}",headers={"Range":f"bytes={lo}-{hi}","Accept-Encoding":"identity"})
        try:
            with urllib.request.urlopen(req,timeout=60) as response:
                if response.status!=206 or not response.headers["Content-Range"].startswith(f"bytes {lo}-{hi}/"):
                    raise ValueError("Server did not return the requested range")
                data=response.read(hi-lo+2)
                if len(data)!=hi-lo+1:
                    raise OSError(f"Incomplete range {lo} ({len(data)} of {hi-lo+1} bytes)")
            path.write_bytes(data)
            return path
        except OSError:
            if attempt==5: raise
            time.sleep(1+attempt)

if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts=list(pool.map(get,range((size+chunk-1)//chunk)))
    out=here/"fast-package.part"
    with out.open("wb") as f:
        for path in parts:f.write(path.read_bytes())
    assert hashlib.sha256(out.read_bytes()).hexdigest()==sha
    os.replace(out,here/"bookworm.deb")
    print("Verified Debian package",sha)
