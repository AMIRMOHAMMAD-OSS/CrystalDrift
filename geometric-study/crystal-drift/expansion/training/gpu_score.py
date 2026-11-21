"""Score exactly the GPU-timed candidates retrospectively on their own targets."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from match import cohort,target_context,candidate,unpack_gram,HERE

GPU=HERE/"gpu"
BUDGETS=(.2,.5,1,2,3,5,8,10,15,20,30,50,80,100,150,250,500,1000,5000,10000)


def one(path):
    ref=cohort()
    out=GPU/(path.name.replace("-episodes.npz","-rms.npz"))
    if out.exists():return
    with np.load(path) as d:
        ids,z,k,f=d["ids"],d["z"],d["k"],d["f"]
    np.testing.assert_array_equal(ids,ref["ids"][:len(ids)])
    rms=np.full(k.shape[:2],np.nan)
    start=time.perf_counter();errors=0
    for i,mid in enumerate(ids):
        ctx=target_context(ref["k"][i],ref["f"][i],z[i])
        active=z[i,z[i]>0].astype(np.int64)
        grams=unpack_gram(k[i])
        for j in range(k.shape[1]):
            try:rms[i,j],_,_=candidate(ctx,k[i,j],f[i,j],active,grams[j],(i+31*j)%107==0)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):
                errors+=1
    tmp=out.with_suffix(".building")
    with tmp.open("wb") as fp:np.savez_compressed(fp,ids=ids,rms=rms,matcher_exceptions=errors)
    tmp.replace(out)
    print(out.name,round(time.perf_counter()-start,1),flush=True)


def table(name,rows):
    with (GPU/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def reports():
    rows=[];cost=[];outcomes={};bootstrap=np.random.default_rng(37111).integers(0,128,(10000,128))
    for p in sorted(GPU.glob("*-rms.npz")):
        method=p.name.split("-rms")[0]
        vol=GPU/f"{method}-10000.npz"
        if not vol.exists():continue
        with np.load(p) as a,np.load(GPU/f"{method}-episodes.npz") as b,np.load(vol) as c:
            np.testing.assert_array_equal(a["ids"],b["ids"])
            t=b["elapsed_ms"];hit=np.isfinite(a["rms"])
            lat=np.diff(t,axis=1,prepend=0).ravel()
            assert np.all(lat>0)
            workload=a["ids"][np.arange(10000)%128]
            np.testing.assert_array_equal(workload,c["ids"])
            assert c["counts"].sum()==10000
            seconds=float(c["elapsed_s"])
            cost.append(dict(method=method,b1_median_ms=np.median(lat),b1_p10_ms=np.quantile(lat,.1),
                b1_p90_ms=np.quantile(lat,.9),seconds_for_10000=seconds,minutes_for_10000=seconds/60,
                structures_per_second=10000/seconds,proposals_timed=len(lat),matcher_exceptions=int(a["matcher_exceptions"])))
            for mode,grid in (("time",BUDGETS),("draws",range(1,21))):
                for k in grid:
                    mask=t<=k if mode=="time" else np.broadcast_to(np.arange(20)<k,t.shape)
                    success=(hit&mask).any(1)
                    outcomes[method,mode,k]=success.astype(float)
                    low,hi=np.quantile(success[bootstrap].mean(1),[.025,.975])
                    rows.append(dict(method=method,axis=mode,budget=k,mean_completed=mask.sum(1).mean(),
                                     targets=len(t),hits=success.sum(),mr=success.mean(),lo=low,hi=hi))
    if rows:
        table("recovery-curves.csv",rows);table("runtime.csv",cost)
        paired=[]
        for (method,mode,budget),success in outcomes.items():
            ref=outcomes.get(("drift",mode,budget))
            if method=="drift" or ref is None:continue
            delta=ref-success
            lo,hi=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            paired.append(dict(first="drift",second=method,axis=mode,budget=budget,
                               targets=len(delta),mr_difference=delta.mean(),lo=lo,hi=hi))
        if paired:table("paired-recovery.csv",paired)
    return len(cost)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--watch",action="store_true");args=ap.parse_args()
    while True:
        for p in sorted(GPU.glob("*-episodes.npz")):one(p)
        if reports()>=7 or not args.watch:break
        time.sleep(5)
