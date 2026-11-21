"""Generate paired unconditional pools from validation-frozen weights only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os
HERE=Path(__file__).resolve().parent
V3=HERE.parent
ROOT=V3.parents[1]
ARMS=(("angular-small","small-angular-laplace-11"),
      ("contacts-medium-long","medium-contacts-laplace-11-8pass"),
      ("angular-medium-long","medium-angular-laplace-11-8pass"))

def main():
    p=argparse.ArgumentParser();p.add_argument("--wait",action="store_true");a=p.parse_args()
    pending=list(ARMS)
    while pending:
        for arm,folder in list(pending):
            base=V3/"models/fits"/folder;reg=base/"selection.json"
            if not reg.exists():continue
            choice=json.loads(reg.read_text());path=base/choice["chosen"]
            if not path.exists():
                pointer=json.loads(path.with_suffix(path.suffix+".pointer.json").read_text())
                path=Path(pointer["windows_path" if os.name=="nt" else "path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==choice["sha256"]
            saved=HERE/"pools"/(arm+".json")
            if not saved.exists():
                subprocess.run([sys.executable,str(HERE/"pool.py"),arm,"--checkpoint",str(path)],cwd=ROOT,check=True)
            assert json.loads(saved.read_text())["ckpt"]==choice["sha256"]
            pending.remove((arm,folder))
        if not a.wait:return
        if pending:time.sleep(15)
if __name__=="__main__":main()
