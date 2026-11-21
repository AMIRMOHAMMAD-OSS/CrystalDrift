"""Fetch the pinned public CrystalFlow inference source files, verifying Git object IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
import time
PIN="9c25ff0245d787efd87c9d1d797a8968443608cc"
URL="https://raw.githubusercontent.com/ixsluo/CrystalFlow/"+PIN
def main(dest):
    tree=requests.get(f"https://api.github.com/repos/ixsluo/CrystalFlow/git/trees/{PIN}?recursive=1",timeout=60).json()["tree"]
    keep=[x for x in tree if x["path"] in ("diffcsp/pl_modules/cspnet.py","diffcsp/pl_modules/flow.py","diffcsp/pl_modules/type_module.py","diffcsp/pl_modules/lattice/matrix.py","diffcsp/pl_modules/ode_solvers.py","scripts/evaluate.py","scripts/generation.py","scripts/sample.py","README.md","diffcsp/common/data_utils.py","LICENSE","diffcsp/pl_modules/lattice_utils.py")]
    def get(x):
        p=dest/x["path"]
        def valid(data):return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()==x["sha"]
        if p.exists() and valid(p.read_bytes()):return
        for i in range(4):
            try:
                data=requests.get(URL+"/"+x["path"],timeout=40).content
                assert valid(data)
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return
            except Exception:
                if i==3:raise
                time.sleep(1)
    with ThreadPoolExecutor(8) as pool:list(pool.map(get,keep))
    print("Verified",len(keep),"source blobs at",PIN)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("dest",type=Path);main(parser.parse_args().dest)
