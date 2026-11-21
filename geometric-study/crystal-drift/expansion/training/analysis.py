"""Rebuild nested full-split match rates and paired intervals from candidate-level records."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHODS=("drift","iid","meanflow","flow1","flow16","retrieval","uflow")
SEEDS=(11,23,37)
BUDGETS=(1,5,20)
METRICS=("mr","rmse","max_site","volume_mr")


def write(path,val):
    tmp=path.with_suffix(".building")
    tmp.write_text(json.dumps(val,indent=1,allow_nan=False)+"\n")
    tmp.replace(path)


def csvfile(path,rows):
    with path.open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def array_metrics(pool,k):
    r=pool["rms"][:,:k]; finite=np.isfinite(r)
    hit=finite.any(1)
    best=np.where(finite,r,np.inf).min(1)
    strict=pool["strict_fit"][:,:k].any(1)
    vol=(finite&(np.abs(pool["log_volume_ratio"][:,:k])<=np.log(1.10))).any(1)
    return np.stack([hit,np.where(hit,best,0.),strict,vol],axis=1)


def estimate(means):
    out=means.copy()
    np.divide(means[...,1],means[...,0],out=out[...,1],where=means[...,0]>0)
    return out


def run(iterations=10000):
    with np.load(ROOT/"revision/full/mp20/complete-crosswalk-k20.npz") as old:
        ids=old["ids"]; arms=old["arms"].tolist()
        retr={}
        for s in SEEDS:
            i=arms.index(f"retrieval-seed{s}")
            retr[s]={k:old[k][:,i] for k in ("rms","strict_fit","log_volume_ratio")}
    with np.load(ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz") as ext:
        np.testing.assert_array_equal(ids,ext["ids"])
        released={k:ext[k] for k in ("rms","strict_fit","log_volume_ratio")}
    iterative=None
    path=HERE.parent/"external/csp-s5/score-crosswalk.npz"
    if path.exists():
        with np.load(path) as ext:
            np.testing.assert_array_equal(ids,ext["ids"])
            if ext["scored"].all():iterative={k:ext[k] for k in ("rms","strict_fit","log_volume_ratio")}
    n=len(ids); reps=len(SEEDS)
    active=[];entries=[];per_seed=[];quality=[]
    for method in METHODS+(("crystalflow5",) if iterative is not None else ()):
        seeds=(2026,) if method=="uflow" else ("release",) if method=="crystalflow5" else SEEDS
        if method not in ("retrieval","uflow","crystalflow5") and not all((HERE/f"test/{method}-seed{s}/outcomes.npz").exists() for s in seeds):
            continue
        active.append(method)
        by_seed=[]
        for s in seeds:
            if method=="retrieval":pool=retr[s]
            elif method=="uflow":pool=released
            elif method=="crystalflow5":pool=iterative
            else:
                with np.load(HERE/f"test/{method}-seed{s}/outcomes.npz") as raw:
                    np.testing.assert_array_equal(ids,raw["ids"])
                    pool={k:raw[k] for k in ("rms","strict_fit","log_volume_ratio")}
            per_budget=[]
            for k in BUDGETS:
                vec=array_metrics(pool,k)
                mean=estimate(vec.mean(0))
                per_seed.append(dict(method=method,seed=s,k=k,hits=int(vec[:,0].sum()),targets=n,
                                     **dict(zip(METRICS,mean.tolist()))))
                per_budget.append(vec)
            by_seed.append(np.stack(per_budget,1))
        if len(by_seed)==1:by_seed*=3
        entries.append(np.stack(by_seed,1))
    # Shape is material, method, fitted seed, draw count, measurement.
    cases=np.stack(entries,1)
    expected=estimate(cases.mean(0)).mean(1)
    rng=np.random.default_rng(753127)
    cached=[]
    for i in range(0,iterations,30):
        weight=rng.multinomial(n,np.full(n,1/n),size=min(30,iterations-i))
        avg=((weight@cases.reshape(n,-1))/n).reshape(-1,*cases.shape[1:])
        cached.extend(estimate(avg).mean(2))
    boot=np.array(cached)
    for m,method in enumerate(active):
        for j,k in enumerate(BUDGETS):
            row=dict(method=method,k=k)
            for c,name in enumerate(METRICS):
                low,hi=np.quantile(boot[:,m,j,c],[.025,.975])
                row[name]=float(expected[m,j,c]);row[f"{name}_lo"]=float(low);row[f"{name}_hi"]=float(hi)
            quality.append(row)
    csvfile(HERE/"test-quality.csv",quality)
    csvfile(HERE/"test-quality-seeds.csv",per_seed)
    if "drift" in active:
        base=active.index("drift");rows=[]
        for m,method in enumerate(active):
            if m==base:continue
            for j,k in enumerate(BUDGETS):
                for c,name in enumerate(METRICS):
                    delta=boot[:,base,j,c]-boot[:,m,j,c]
                    low,high=np.quantile(delta,[.025,.975])
                    rows.append(dict(a="drift",b=method,k=k,metric=name,
                        mean=float(expected[base,j,c]-expected[m,j,c]),lo=float(low),hi=float(high)))
        csvfile(HERE/"test-paired.csv",rows)
    write(HERE/"analysis.json",dict(models=active,cases=n,seeds=SEEDS,bootstrap=iterations,
        bootstrap_seed=753127,matched_rms_denominator="recovered targets within each fitted seed, then arithmetic mean over seeds",
        complete=all(m in active for m in METHODS),released_iterative_complete=iterative is not None))
    print(active,iterations,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates",type=int,default=10000)
    run(ap.parse_args().replicates)
