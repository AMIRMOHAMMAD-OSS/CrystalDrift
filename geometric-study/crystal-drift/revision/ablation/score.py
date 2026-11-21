"""Score fixed common ensembles and retain per-candidate geometry and matcher outcomes."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
sys.path.insert(0,str(STUDY))
import data
from standard_match import SETTINGS, make_structure, order

ARMS=("pair","original","harmonic","reweight")
SEEDS=(11,23,37)
METRICS=("forecast_energy","fair_energy","contact_valid","volume_error","match1","match5")


def load(path):
    with np.load(path,allow_pickle=False) as f:
        return {key:f[key] for key in ("k","f","z","ids")}


def assess(arm,seed,split,ref,observed,scale,selected):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    out=HERE/"runs"/arm
    casefile=out/f"seed{seed}-{split}-cases.csv"
    if casefile.exists():
        return
    bank=load(out/f"seed{seed}-{split}.npz")
    x,f,z,ids=(bank[name] for name in ("k","f","z","ids"))
    assert np.array_equal(ids,ref["ids"]) and np.array_equal(z,ref["z"])
    n=len(ids);k=5
    a=data.geometry_features(x.reshape(-1,6),f.reshape(-1,8,3),np.repeat(z,k,0)).reshape(n,k,4)
    ok=np.isfinite(a).all(-1)
    y=np.log(np.maximum(observed,1e-8))/scale
    v=np.log(np.maximum(a,1e-8))/scale
    # Keep all finite draws in forecasts including physically short contacts.
    values=[]
    for i,row in enumerate(v):
        accepted=row[ok[i]]
        if not len(accepted): raise ValueError(f"No finite draw for {ids[i]}")
        errors=np.linalg.norm(accepted-y[i],axis=1).mean()
        spread=np.linalg.norm(accepted[:,None]-accepted[None],axis=-1).sum()/2
        m=len(accepted)
        contact=ok[i] & (np.min(a[i,:,1:],axis=-1)>=.5)
        values.append([errors-spread/m**2, errors-spread/(m*(m-1)) if m>=2 else np.nan,
                       contact.mean(),np.abs(np.log(a[i,ok[i],0]/observed[i,0])).mean(),np.nan,np.nan])
    distance=np.full((n,k),np.nan);threw=0
    matcher=StructureMatcher(**SETTINGS)
    for i in selected:
        truth=make_structure(ref["k"][i],ref["f"][i],z[i])
        for j in range(k):
            try:
                match=matcher.get_rms_dist(make_structure(x[i,j],f[i,j],z[i]),truth)
                if match is not None: distance[i,j]=match[0]
            except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError):
                threw+=1
        hit=np.isfinite(distance[i])
        values[i][-2:]=[int(hit[0]),int(hit.any())]
    np.savez_compressed(out/f"seed{seed}-{split}-observables.npz",ids=ids,observed=observed,
                        predicted=a,evaluable=ok,matcher_indices=selected,rms_by_candidate=distance[selected])
    with casefile.open("w",newline="",encoding="utf8") as stream:
        w=csv.writer(stream);w.writerow(["id",*METRICS])
        for mid,vals in zip(ids,values):
            w.writerow([mid,*[v if np.isfinite(v) else "" for v in vals]])
    print(split,arm,seed,"ES",round(np.mean(np.asarray(values)[:,0]),4),"@5",
          round(np.nanmean(np.asarray(values)[:,-1]),4),"numerical failures",int((~ok).sum()),"matcher exceptions",threw,flush=True)
    (out/f"seed{seed}-{split}-exceptions.json").write_text(json.dumps(
        {"unevaluable_draws":int((~ok).sum()),"matcher_exceptions":threw}),encoding="utf8")


def summarize(split):
    values={}
    seeds=[]
    for arm in ARMS:
        cases=[]
        for seed in SEEDS:
            with (HERE/"runs"/arm/f"seed{seed}-{split}-cases.csv").open(encoding="utf8") as stream:
                table=list(csv.DictReader(stream))
            row=np.array([[float(record[key]) if record[key] else np.nan for key in METRICS] for record in table])
            cases.append(row)
            seeds.append(dict(arm=arm,seed=seed,**dict(zip(METRICS,np.nanmean(row,axis=0).tolist()))))
        values[arm]=np.mean(cases,axis=0)
    contrasts=[("harmonic","original"),("original","pair"),("harmonic","pair"),
               ("harmonic","reweight"),("reweight","original")]
    summary={}
    for index,key in enumerate(METRICS):
        valid=np.logical_and.reduce([np.isfinite(rows[:,index]) for rows in values.values()])
        n=int(valid.sum()); gen=np.random.default_rng(741123)
        blocks=gen.multinomial(n,np.full(n,1/n),size=10000).astype(np.float64)
        results={}
        def interval(v):
            sample=blocks@v/n
            return dict(mean=float(v.mean()),ci95=np.quantile(sample,[.025,.975]).tolist())
        for arm,vals in values.items():
            results[arm]=interval(vals[valid,index])
        for later,earlier in contrasts:
            results[later+"_minus_"+earlier]=interval(values[later][valid,index]-values[earlier][valid,index])
        summary[key]=dict(n=n,estimates=results)
    (HERE/f"{split}-summary.json").write_text(json.dumps(dict(paired_by_case=summary,per_seed=seeds),indent=1)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site",type=Path,nargs="*",default=[])
    options=parser.parse_args();sys.path.extend(map(str,options.site))
    training=load(STUDY/"data/multi-train.npz")
    h=data.geometry_features(training["k"],training["f"],training["z"])
    scale=np.log(np.maximum(h,1e-8)).std(axis=0);scale[scale<1e-8]=1
    for part in ("val","test"):
        ref=load(STUDY/"data"/f"multi-{part}.npz")
        obs=data.geometry_features(ref["k"],ref["f"],ref["z"])
        idx=order("multi",ref["ids"])
        for seed in SEEDS:
            for arm in ARMS:
                deadline=time.monotonic()+900
                while not (HERE/"runs"/arm/f"seed{seed}.json").is_file():
                    if time.monotonic()>deadline:raise TimeoutError(arm)
                    time.sleep(2)
                assess(arm,seed,part,ref,obs,scale,idx)
        summarize(part)
