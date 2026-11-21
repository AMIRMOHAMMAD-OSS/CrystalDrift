"""Resume a complete local run from the last finished stage. Invoke in a Torch environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ARMS=("radial","angular","power0","fixed-angular","flow16")
SEEDS=(11,23,37)


def call(script,*args,python=sys.executable):
    subprocess.run([python,str(HERE/script),*map(str,args)],check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--matcher-python",required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--watch-existing-fits",action="store_true")
    args=p.parse_args()
    if not (HERE/"prior.npz").exists():call("model.py")
    if not (HERE/"cache.json").exists():call("cache.py")
    if not (HERE/"overfit.json").exists():call("stage.py","--overfit")
    if not args.watch_existing_fits:
        call("train.py","--seeds",11,"--stop",256,"--workers",1)
        call("stage.py","--seeds",11,"--updates",16,64,256)
        call("train.py","--stop",3392,"--workers",1)
    while True:
        finished=[]
        for arm in ARMS:
            if not all((HERE/f"fits/{arm}-{s}/update-03392.json").is_file() for s in SEEDS):continue
            if (HERE/"selection.json").exists() and arm in json.loads((HERE/"selection.json").read_text()):
                finished.append(arm);continue
            call("stage.py","--arms",arm,"--updates",256,1024,3392)
            call("lock.py","--arms",arm)
            finished.append(arm)
        if len(finished)==len(ARMS):break
        time.sleep(20)
    call("stage.py","--arms","prior","--updates",0)
    call("predict.py","--arms","prior",*ARMS)
    call("match.py","--workers",args.workers,python=args.matcher_python)

if __name__=="__main__":main()
