"""Resume a registered generation/score matrix by panel; retain paired shards on interruption."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument("matrix",type=Path);p.add_argument("--panels",nargs="+",required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--inner",type=int,default=32)
    p.add_argument("--matcher");p.add_argument("--mode",choices=("predict","score"),required=True);p.add_argument("--workers",type=int,default=1)
    p.add_argument("--after",type=Path);p.add_argument("--timeout-hours",type=float,default=12);p.add_argument("--draws",type=int,default=20)
    a=p.parse_args();start=time.monotonic()
    while a.after and not a.after.exists():
        if time.monotonic()-start>3600*a.timeout_hours:raise TimeoutError(str(a.after))
        time.sleep(10)
    specs=json.loads(a.matrix.read_text())["methods"]
    for panel in a.panels:
        if a.mode=="predict":cmd=[sys.executable,str(HERE/"paired.py"),str(a.matrix),"--panel",panel,"--draws",str(a.draws),"--device",a.device,"--inner",str(a.inner)]
        else:cmd=[a.matcher or sys.executable,str(HERE/"shards.py"),panel,*[s["tag"] for s in specs],"--workers",str(a.workers),"--wait"]
        print("Running",cmd,flush=True)
        subprocess.run(cmd,check=True,cwd=ROOT)
if __name__=="__main__":main()
