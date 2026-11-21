"""Retrospectively score balanced episodes, preserving owned elapsed times and per-ID uncertainty."""
import csv
from pathlib import Path
import numpy as np
from gpu_score import original,HERE
DST=HERE/"interleaved";original.GPU=DST
if __name__=="__main__":
    rows=[];paired=[];hits={};boot=np.random.default_rng(381075).integers(0,128,(5000,128))
    for p in sorted(DST.glob("*-episodes.npz")):
        original.one(p)
        m=p.name[:-len("-episodes.npz")]
        with np.load(p) as t,np.load(DST/f"{m}-rms.npz") as score:
            np.testing.assert_equal(t["ids"],score["ids"]);ms=t["elapsed_ms"];r=score["rms"]
            for budget in (1,2,5,10,20,30,50,80,100,150,250,500,1000,2500,5000):
                done=ms<=budget;yes=(done&np.isfinite(r)).any(1);hits[m,budget]=yes.astype(float)
                lo,hi=np.quantile(yes[boot].mean(1),[.025,.975])
                rows.append(dict(method=m,ms=budget,targets=128,completed=done.sum(1).mean(),hits=int(yes.sum()),mr=yes.mean(),
                         low=lo,high=hi,b1_median_ms=np.median(np.diff(ms,axis=1,prepend=0))))
    for (m,b),h in hits.items():
        dif=hits["angular",b]-h;lo,hi=np.quantile(dif[boot].mean(1),[.025,.975])
        paired.append(dict(first="angular",second=m,ms=b,mr_delta=dif.mean(),low=lo,high=hi))
    for name,data in (("recovery.csv",rows),("paired.csv",paired)):
        with (DST/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
