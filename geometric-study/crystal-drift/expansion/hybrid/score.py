"""Match all confirmation draws and calculate within-ID hybrid rescues and losses."""
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from time import perf_counter
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[key]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"revision/full"),str(ROOT)]
from match_full import candidate,target_context
from data import unpack_gram
from concurrent.futures import ProcessPoolExecutor
METHODS=("drift","flow1","flow16","retrieval")
SEEDS=(11,23,37)
ARMS=[f"{m}-seed{s}" for m in METHODS for s in SEEDS]


@lru_cache(maxsize=1)
def resources():
    with np.load(HERE/"cohort.npz") as d:ref={k:d[k] for k in ("z","f","k","ids")}
    order=None;pools=[]
    for arm in ARMS:
        with np.load(HERE/f"pools/{arm}-confirm.npz") as p:
            if order is None:
                lut={mid:i for i,mid in enumerate(ref["ids"])}
                order=np.array([lut[mid] for mid in p["ids"]])
                ref={k:v[order] for k,v in ref.items()}
            np.testing.assert_array_equal(ref["ids"],p["ids"])
            np.testing.assert_array_equal(ref["z"],p["z"])
            pools.append((p["k"],p["f"]))
    return ref,pools


def score(lo):
    path=HERE/f"tile-{lo:04d}.npz"
    if path.exists():return
    ref,pools=resources();hi=min(lo+16,len(ref["ids"]))
    rms=np.full((hi-lo,len(ARMS),20),np.nan);errors=0
    for i in range(lo,hi):
        z=ref["z"][i];ctx=target_context(ref["k"][i],ref["f"][i],z)
        live=z[z>0].astype(np.int64);memo={}
        for arm,(ks,fs) in enumerate(pools):
            grams=unpack_gram(ks[i])
            for draw in range(20):
                k,f=ks[i,draw],fs[i,draw]
                key=(k.tobytes(),f.tobytes())
                if key not in memo:
                    try:
                        memo[key]=candidate(ctx,k,f,live,grams[draw],i%64==0 and draw==0)[0]
                    except (ValueError,OverflowError,np.linalg.LinAlgError):
                        errors+=1;memo[key]=np.nan
                rms[i-lo,arm,draw]=memo[key]
    with path.with_suffix(".part").open("wb") as stream:
        np.savez_compressed(stream,rms=rms,ids=ref["ids"][lo:hi],exceptions=errors,arms=ARMS)
    path.with_suffix(".part").replace(path)


def summarize(name,ids,rms):
    rng=np.random.default_rng(129786)
    hit=np.isfinite(rms).reshape(len(ids),4,3,20)
    r1=hit[:,3,:,0];r20=hit[:,3].any(-1)
    stats={};rows=[];perseed=[]
    for j,m in enumerate(METHODS[:3]):
        n19=hit[:,j,:,:19].any(-1);mix=r1|n19;net=hit[:,j].any(-1)
        measures=dict(retrieved_1=r1,retrieved_20=r20,generated_20=net,hybrid=mix,
            rescued_first=(~r1)&n19,rescued_all20=(~r20)&mix,lost_vs_all20=r20&(~mix),
            difference_vs_retrieval=mix.astype(float)-r20,difference_vs_neural=mix.astype(float)-net)
        for key,mask in measures.items():
            avg=mask.mean(1);n=len(ids)
            boot_rng=np.random.default_rng(129786)
            draws=np.concatenate([avg[boot_rng.integers(0,n,(100,n))].mean(1) for b in range(100)])
            stats.setdefault(m,{})[key]=[float(avg.mean()),*np.quantile(draws,[.025,.975]).tolist()]
        for s,seed in enumerate(SEEDS):
            perseed.append(dict(method=m,seed=seed,**{key:float(v[:,s].mean()) for key,v in measures.items()}))
            rows.extend(dict(id=str(mid),method=m,seed=seed,**{key:float(v[i,s]) for key,v in measures.items()}) for i,mid in enumerate(ids))
    for label,values in (("per-case",rows),("per-seed",perseed)):
        with (HERE/f"{name}-{label}.csv").open("w",newline="") as file:
            w=csv.DictWriter(file,fieldnames=values[0]);w.writeheader();w.writerows(values)
    (HERE/f"{name}.json").write_text(json.dumps(dict(n=len(ids),seeds=SEEDS,results=stats),indent=2)+"\n")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(score,range(0,512,16)))
    with np.load(ROOT/"revision/full/mp20/complete-crosswalk-k20.npz") as d:
        np.testing.assert_array_equal(d["arms"],ARMS)
        summarize("exploratory",d["ids"],d["rms"])
    tiles=[np.load(HERE/f"tile-{i:04d}.npz") for i in range(0,512,16)]
    ids=np.concatenate([t["ids"] for t in tiles]);rms=np.concatenate([t["rms"] for t in tiles])
    assert len(set(ids))==512
    np.savez_compressed(HERE/"confirmation-scores.npz",ids=ids,rms=rms,arms=ARMS,exceptions=sum(int(t["exceptions"]) for t in tiles))
    summarize("confirmation",ids,rms)
