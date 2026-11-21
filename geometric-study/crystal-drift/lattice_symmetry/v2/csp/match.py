"""Reuse the baseline's pinned primitive matcher, certified shape prescreen and reductions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import time
for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[v]="1"
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location("fixed_matching",ROOT/"expansion/training/match.py")
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
evaluator.HERE=HERE


def one(job):
    evaluator.score(job)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=3)
    p.add_argument("--watch",action="store_true");p.add_argument("--arms",nargs="*")
    p.add_argument("--assemble-only",action="store_true",help="Join complete score shards without scheduling predictions.")
    p.add_argument("--reverse",action="store_true",help="Visit pending shards from the end for a second independent worker pool.")
    p.add_argument("--expected",type=int,default=18);args=p.parse_args()
    assert evaluator.version("pymatgen")=="2024.4.13"
    if args.assemble_only:
        for ready in sorted((HERE/"test").glob("*/ready.json")):
            if not (ready.parent/"outcomes.npz").exists():evaluator.assemble(ready.parent)
        sys.exit(0)
    with (ProcessPoolExecutor(args.workers) if args.workers>1 else nullcontext()) as pool:
        while True:
            complete=0;jobs=[]
            for mark in sorted((HERE/"test").glob("*/ready.json")):
                base=mark.parent
                if args.arms and base.name not in args.arms:continue
                if (base/"outcomes.npz").exists() or evaluator.assemble(base):complete+=1;continue
                scores=base/"scores";scores.mkdir(exist_ok=True)
                for i in range(0,9046,64):
                    j=min(i+64,9046)
                    if not (scores/f"{i:05}-{j:05}.npz").exists():jobs.append((base.name,i,j))
            if jobs:
                if args.reverse:jobs.reverse()
                completed=pool.map(one,jobs) if pool is not None else map(one,jobs)
                for _ in completed:pass
                continue
            if not args.watch or complete==(len(args.arms) if args.arms else args.expected):break
            time.sleep(12)
