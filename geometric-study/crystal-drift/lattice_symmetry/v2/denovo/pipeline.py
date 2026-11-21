"""Resume the primary full-chemistry banks from the corresponding locked CSP selections."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LOCK=HERE.parent/"csp/selection.json"


def run(python,script,*args,env):
    command=[str(python),str(HERE/script),*map(str,args)]
    print("Running",script,*args,flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torch-python",type=Path,default=Path(sys.executable))
    p.add_argument("--matcher-python",type=Path,required=True)
    p.add_argument("--metrics-site",type=Path,required=True)
    p.add_argument("--wait",action="store_true")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--screen-workers",type=int,default=1)
    a=p.parse_args();env=os.environ.copy();env["CRYSTAL_METRICS_SITE"]=str(a.metrics_site.resolve())
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[key]="1"
    arms=["radial","angular"]
    for arm in arms:
        while not (LOCK.exists() and arm in json.loads(LOCK.read_text())):
            if not a.wait:raise SystemExit(f"Still awaiting locked {arm} selection")
            sleep(15)
        # Completion clock is written after the sealed ordered pool and batch-one observations.
        if not (HERE/"pools"/(arm+"-cpu.json")).is_file():
            run(a.torch_python,"generate_pools.py",arm,env=env)
        for stage in ("raw","fp","novelty","report"):
            run(a.matcher_python,"evaluate.py",stage,"--arms",arm,"--workers",a.workers,env=env)
    run(a.matcher_python,"screen.py","--workers",a.screen_workers,env=env)


if __name__=="__main__":main()
