"""Download the authors' MP-20 CSP weight snapshot using independently checked byte ranges."""
import concurrent.futures
import os
from pathlib import Path
import sys
import time
import zipfile
import hashlib
import requests

ID="1XlV0kntXqkyVoU-wslGQVa28852v2Hpv"
URL="https://drive.usercontent.google.com/download"
def fetch(lo,hi):
    last=None
    for i in range(5):
        try:
            r=requests.get(URL,params={"id":ID,"export":"download","confirm":"t","range_start":lo,"try":i},
                           headers={"Range":f"bytes={lo}-{hi}"},timeout=(15,80),stream=True)
            if r.status_code!=206 or not r.headers.get("Content-Range","").startswith(f"bytes {lo}-"):
                raise ValueError((r.status_code,r.url,r.headers.get("Content-Range"),r.headers.get("Content-Type")))
            body=r.content; total=int(r.headers["Content-Range"].split("/")[-1])
            if len(body)!=min(hi,total-1)-lo+1:raise IOError(f"truncated range {lo}")
            return body,total
        except Exception as e:last=e
    raise last

def main(dest):
    first,total=fetch(0,1023)
    print("Total",total,flush=True);chunks=dest.parent/"diffpp-pieces";chunks.mkdir(exist_ok=True)
    size=2<<20
    def part(lo):
        p=chunks/f"{lo:09}"
        hi=min(total,lo+size)-1
        if not p.exists() or p.stat().st_size!=hi-lo+1:
            body,expected=fetch(lo,hi);assert expected==total
            p.write_bytes(body)
        return p
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        paths=list(pool.map(part,range(0,total,size)))
    tmp=dest.with_suffix(".tmp")
    with tmp.open("wb") as ff:
        for p in paths:ff.write(p.read_bytes())
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    os.replace(tmp,dest)
    print("sha256",hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
if __name__=="__main__":main(Path(sys.argv[1]))
