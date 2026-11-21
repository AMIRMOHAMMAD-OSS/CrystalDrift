"""Score GPU replay tiles with the shared full-split matcher and emit casewise results."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[var]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/"mp20"
DONE=OUT/"matches"
PARTS=OUT/"intermediate-matches"
sys.path.insert(0,str(HERE.parent))
from match_full import candidate,target_context,unpack_gram,pin,SETTINGS

REF=None
def init():
    global REF
    pin()
    with np.load(HERE.parent/"mp20/cohort.npz") as f:
        REF={k:f[k] for k in ("ids","k","f","z")}


def inputpath(lo,hi,d):
    return OUT/f"draw{d:02}-cases{lo:05}-{hi:05}.npz"


def outpath(job):
    lo,hi,first,last=job
    directory=PARTS if first>=5 and (first,last)!=(5,20) else DONE
    return directory/f"cases{lo:05}-{hi:05}-draws{first:02}-{last:02}.npz"


def score(job):
    lo,hi,first,last=job; num=last-first
    if outpath(job).exists():return job
    values=[];positions=[]
    for draw in range(first,last):
        with np.load(inputpath(lo,hi,draw),allow_pickle=False) as t:
            np.testing.assert_array_equal(REF["ids"][lo:hi],t["ids"])
            np.testing.assert_array_equal(REF["z"][lo:hi],t["z"])
            values.append(t["k"]);positions.append(t["f"])
    ks=np.stack(values,1);fs=np.stack(positions,1)
    rms=np.full((hi-lo,num),np.nan);strict=np.zeros_like(rms,bool)
    logv=ks[:,:,:3].sum(-1)-REF["k"][lo:hi,:3].sum(-1,dtype=float)[:,None]
    exceptions=Counter(); verified=0; start=time.monotonic()
    for i,index in enumerate(range(lo,hi)):
        z=REF["z"][index];ctx=target_context(REF["k"][index],REF["f"][index],z)
        gg=unpack_gram(ks[i]);zz=z[z>0].astype(np.int64)
        for j in range(num):
            verify=((index*104729+(first+j)*1543)%1031==0)
            try:
                rms[i,j],strict[i,j],_=candidate(ctx,ks[i,j],fs[i,j],zz,gg[j],verify)
            except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as exc:
                exceptions[type(exc).__name__]+=1
            verified+=int(verify)
    name=outpath(job);tmp=name.with_suffix(".part.npz")
    np.savez_compressed(tmp,ids=REF["ids"][lo:hi],rms=rms,strict_fit=strict,
                        log_volume_ratio=logv,first=first,last=last,public_api_checks=verified,
                        exceptions=json.dumps(exceptions),seconds=time.monotonic()-start)
    tmp.replace(name)
    return job


def merge_late(lo,hi):
    destination=outpath((lo,hi,5,20))
    if destination.exists():return
    parts=[outpath((lo,hi,a,a+5)) for a in (5,10,15)]
    if not all(p.is_file() for p in parts):return
    fields={k:[] for k in ("rms","strict_fit","log_volume_ratio")}
    checks=0;seconds=0.;errors=Counter();ids=None
    for p in parts:
        with np.load(p) as d:
            if ids is None:ids=d["ids"]
            else:np.testing.assert_array_equal(ids,d["ids"])
            for key in fields:fields[key].append(d[key])
            checks+=int(d["public_api_checks"]);seconds+=float(d["seconds"])
            errors.update(json.loads(str(d["exceptions"])))
    tmp=destination.with_suffix(".part.npz")
    np.savez_compressed(tmp,ids=ids,first=5,last=20,public_api_checks=checks,exceptions=json.dumps(errors),
                        seconds=seconds,**{key:np.concatenate(values,axis=1) for key,values in fields.items()})
    tmp.replace(destination)


def report():
    with np.load(HERE.parent/"mp20/cohort.npz") as f: ids=f["ids"]
    n=len(ids);rms=np.full((n,20),np.nan);strict=np.zeros((n,20),bool);vol=np.full((n,20),np.nan)
    occupied=np.zeros((n,20),bool);ix={v:i for i,v in enumerate(ids)}
    for f in DONE.glob("cases*.npz"):
        if f.name.endswith(".part.npz"):continue
        with np.load(f) as d:
            rows=[ix[x] for x in d["ids"]];a=int(d["first"]);b=int(d["last"])
            rms[rows,a:b]=d["rms"];strict[rows,a:b]=d["strict_fit"];vol[rows,a:b]=d["log_volume_ratio"];occupied[rows,a:b]=True
    summary={}
    for k in (1,5,20):
        complete=occupied[:,:k].all(1)
        if not complete.any():continue
        r=rms[complete,:k];h=np.isfinite(r)
        record=dict(targets=int(complete.sum()),hits=int(h.any(1).sum()),max_site_hits=int(strict[complete,:k].any(1).sum()),
                    same_draw_native_hits=int((h&(np.abs(vol[complete,:k])<=np.log(1.10))).any(1).sum()))
        record["mean_best_normalized_rms_among_hits"]=float(np.nanmin(r[h.any(1)],axis=1).mean())
        summary[str(k)]=record
    tmp=OUT/"progress.part.json";tmp.write_text(json.dumps(summary,indent=2)+"\n");tmp.replace(OUT/"progress.json")
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=2)
    p.add_argument("--initial-workers",type=int);args=p.parse_args()
    DONE.mkdir(exist_ok=True,parents=True);PARTS.mkdir(exist_ok=True,parents=True)
    groups=[(0,128)]+[(i,min(i+256,9046)) for i in range(128,9046,256)]
    for lo,hi in groups:merge_late(lo,hi)
    jobs=[(lo,hi,start,end) for start,end in ((0,1),(1,5),(5,10),(10,15),(15,20))
          for lo,hi in groups if start<5 or not outpath((lo,hi,5,20)).exists()]
    pending=[j for j in jobs if not outpath(j).exists()]
    active={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
        count=0
        while pending or active:
            slots=args.initial_workers or args.workers
            override=OUT/"matcher-workers.txt"
            if override.is_file():slots=int(override.read_text().strip())
            slots=max(1,min(args.workers,slots))
            ready=[j for j in pending if all(inputpath(j[0],j[1],d).exists() for d in range(j[2],j[3]))]
            for job in ready[:max(0,slots-len(active))]:
                active[pool.submit(score,job)]=job;pending.remove(job)
            if not active:
                time.sleep(3);continue
            done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                job=future.result();active.pop(future);count+=1
                if job[2]>=5:merge_late(job[0],job[1])
            if done and (count%5==0 or not pending):
                print("finished tile groups",count,report(),flush=True)
    print("final",report(),flush=True)

if __name__=="__main__":
    main()
