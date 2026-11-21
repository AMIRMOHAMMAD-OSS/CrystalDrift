"""Run the released DiffCSP/novelty evaluator with isolated v2 output paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/"expansion/denovo"
sys.path.insert(0,str(OLD))
import score
import novelty
import report

score.HERE = novelty.HERE = report.HERE = HERE
native_tiles = report.load_tiles


def tiles(name, kind):
    if name != "reference":
        return native_tiles(name,kind)
    # Only this read targets the immutable release's saved fingerprint tiles.
    outdir = report.HERE
    try:
        report.HERE = OLD
        return native_tiles(name,kind)
    finally:
        report.HERE = outdir


report.load_tiles = tiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("raw","fp","novelty","report"))
    p.add_argument("--arms",nargs="+",default=["angular","radial"])
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--chunk",type=int)
    args = p.parse_args()
    if args.mode == "report":
        sys.argv = ["report","--arms",*args.arms]
        report.main()
    elif args.mode == "novelty":
        sys.argv = ["novelty",*args.arms,"--workers",str(args.workers)]
        novelty.main()
    else:
        if args.mode=="fp":
            os.environ["CRYSTAL_COMPILED_FINGERPRINT"]="1"
        chunk = args.chunk or (20 if args.mode=="fp" else 100)
        jobs=[]
        for name in args.arms:
            n=len(score.arrays(name)["ids"])
            jobs.extend((name,i,min(n,i+chunk),args.mode) for i in range(0,n,chunk)
                        if not (HERE/"scores"/args.mode/name/f"{i:05}-{min(n,i+chunk):05}.npz").exists())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,_ in enumerate(pool.map(score.worker,jobs),1):
                if i%20==0 or i==len(jobs):print(i,len(jobs),flush=True)


if __name__=="__main__":
    main()
