"""Score validation-selected candidates in resumable shards using the verified matching code."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"revision/full"))
from match_full import candidate,target_context
from data import unpack_gram


@lru_cache(maxsize=1)
def cohort():
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:return {k:d[k] for k in ("ids","z","k","f")}


@lru_cache(maxsize=16)
def arrays(tag):
    base=HERE/"test"/tag
    return tuple(np.load(base/(v+".npy"),mmap_mode="r") for v in ("k","f"))


def score(job):
    tag,lo,hi=job
    out=HERE/"test"/tag/"scores";tgt=out/f"{lo:05}-{hi:05}.npz"
    if tgt.exists():return
    ref=cohort(); ks,fs=arrays(tag)
    rms=np.full((hi-lo,20),np.nan)
    fit=np.zeros((hi-lo,20),bool);volume=np.empty((hi-lo,20),np.float32)
    nverify=0;status=np.zeros((hi-lo,20),np.int8)
    forwarded=0
    tic=time.perf_counter()
    for i in range(lo,hi):
        z0=ref["z"][i]
        z=z0[z0>0].astype(np.int64)
        ctx=target_context(ref["k"][i],ref["f"][i],z0)
        k=ks[i];f=fs[i];gg=unpack_gram(k);j=i-lo
        volume[j]=np.sum(k[:,:3],axis=1,dtype=float)-sum(ref["k"][i,:3].astype(float))
        for b in range(20):
            verify=(i*104729+b*1543)%6353==0
            try:
                rms[j,b],fit[j,b],sent=candidate(ctx,k[b],f[b],z,gg[b],verify)
                nverify+=verify;forwarded+=sent
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
                status[j,b]=1
    temp=tgt.with_suffix(f".{os.getpid()}.building")
    with temp.open("wb") as fp:
        np.savez_compressed(fp,ids=ref["ids"][lo:hi],rms=rms,strict_fit=fit,log_volume_ratio=volume,
            error=status,public_checks=nverify,forwarded=forwarded,elapsed_seconds=time.perf_counter()-tic)
    temp.replace(tgt)


def assemble(base):
    rows=[]
    paths=sorted((base/"scores").glob("*.npz"))
    if not paths:return False
    last=0
    for path in paths:
        lo,hi=map(int,path.stem.split("-"))
        if lo!=last:return False
        last=hi
    if last!=9046:return False
    for p in paths:
        with np.load(p) as d:rows.append({k:d[k] for k in d})
    keys=("ids","rms","strict_fit","log_volume_ratio","error")
    out={k:np.concatenate([row[k] for row in rows]) for k in keys}
    for k in ("public_checks","forwarded","elapsed_seconds"):
        out[k]=sum(row[k] for row in rows)
    assert len(set(out["ids"]))==9046
    np.testing.assert_array_equal(out["ids"],cohort()["ids"])
    temp=base/f"outcomes-{os.getpid()}.building"
    with temp.open("wb") as fp:np.savez_compressed(fp,**out)
    temp.replace(base/"outcomes.npz")
    print(base.name,"scored",int(np.isfinite(out["rms"]).any(1).sum()),"hits",int(out["error"].sum()),"exceptions",flush=True)
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",type=int,default=5)
    ap.add_argument("--chunk",type=int,default=64)
    ap.add_argument("--watch",action="store_true")
    ap.add_argument("--assemble-only",action="store_true",help="Join fully scored shards without scheduling work.")
    ap.add_argument("--arms",nargs="+",help="Score only these fitted method-seed names.")
    args=ap.parse_args()
    assert version("pymatgen")=="2024.4.13"
    if args.assemble_only:
        for ready in sorted((HERE/"test").glob("*/ready.json")):
            if not (ready.parent/"outcomes.npz").exists():assemble(ready.parent)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            jobs=[]
            count=0
            for path in sorted((HERE/"test").glob("*/ready.json")):
                base=path.parent
                if args.arms and base.name not in args.arms:continue
                if (base/"outcomes.npz").is_file():
                    count+=1;continue
                if assemble(base):
                    count+=1;continue
                scores=base/"scores";scores.mkdir(exist_ok=True)
                for lo in range(0,9046,args.chunk):
                    hi=min(9046,lo+args.chunk)
                    if not (scores/f"{lo:05}-{hi:05}.npz").exists():jobs.append((base.name,lo,hi))
            if jobs:
                for ans in as_completed([pool.submit(score,j) for j in jobs]):ans.result()
                continue
            if count==(len(args.arms) if args.arms else 15) or not args.watch:break
            time.sleep(6)


if __name__=="__main__":main()
