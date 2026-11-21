"""Paired fixed-weight analysis from complete ordered per-target crosswalks."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
SPLITS=json.loads((V3/"protocol/splits.json").read_text())

def load(path,ids):
    with np.load(path,allow_pickle=False) as d:
        idx={str(x):i for i,x in enumerate(d["ids"])}
        assert len(idx)==len(d["ids"]) and set(ids)<=idx.keys()
        take=[idx[x] for x in ids];r=d["rms"][take]
        v=d["log_volume_ratio"][take] if "log_volume_ratio" in d else np.full_like(r,np.nan)
        error=(d["error"][take]!=0) if "error" in d else np.zeros_like(r,bool)
        st=d["strict_fit"][take] if "strict_fit" in d else np.zeros_like(r,bool)
        assert not np.any(error & np.isfinite(r))
        return {"r":r,"v":v,"strict":st,"err":error}

def restricted_best(r):
    return np.min(np.where(np.isfinite(r),r,np.inf),axis=-1)

def ci(x,clusters=None):
    x=np.asarray(x,float);rng=np.random.default_rng(7331)
    if clusters is None:
        v,n=np.unique(x,return_counts=True)
        avg=(rng.multinomial(len(x),n/len(x),size=5000)@v)/len(x)
    else:
        _,inv=np.unique(clusters,return_inverse=True)
        totals=np.bincount(inv,weights=x);counts=np.bincount(inv);n=len(totals)
        b=rng.integers(n,size=(5000,n));avg=totals[b].sum(1)/counts[b].sum(1)
    return np.quantile(avg,[.025,.975]).tolist()

def metrics(sets,sl,k):
    best=restricted_best(np.stack([s["r"][sl,:k] for s in sets]))
    hit=np.isfinite(best);native=np.stack([(np.isfinite(s["r"][sl,:k])&(np.abs(s["v"][sl,:k])<=np.log(1.1))).any(1) for s in sets])
    strict=np.stack([s["strict"][sl,:k].any(1) for s in sets])
    vr=np.concatenate([s["v"][sl,:k].ravel() for s in sets]);vr=vr[np.isfinite(vr)]
    rms=[float(b[np.isfinite(b)].mean()) if np.isfinite(b).any() else None for b in best]
    return dict(n=len(best[0]),seeds=len(sets),hits_per_seed=hit.sum(1).tolist(),mr=float(hit.mean()),
                rms_matched=float(np.mean([v for v in rms if v is not None])) if any(v is not None for v in rms) else None,
                native_hits=native.sum(1).tolist(),native=float(native.mean()),strict=float(strict.mean()),
                median_log_v=float(np.median(vr)) if len(vr) else None,
                errors=int(sum(s["err"][sl,:k].sum() for s in sets)),rms_by_seed=rms), (hit,best,native)

def contrasts(a,b,cluster):
    m=max(len(a[0]),len(b[0]));assert len(a[0]) in (1,m) and len(b[0]) in (1,m)
    ha,ba,na=[np.broadcast_to(x,(m,x.shape[-1])) for x in a]
    hb,bb,nb=[np.broadcast_to(x,(m,x.shape[-1])) for x in b]
    both=ha&hb
    d=ha.astype(float)-hb;dn=na.astype(float)-nb
    return dict(first_only=(ha&~hb).sum(1).tolist(),second_only=(hb&~ha).sum(1).tolist(),
                pp=100*d.mean(),ci95_pp=(100*np.array(ci(d.mean(0),cluster))).tolist(),seed_pp=(100*d.mean(1)).tolist(),
                native_pp=100*dn.mean(),native_ci95_pp=(100*np.array(ci(dn.mean(0),cluster))).tolist(),
                common_counts=both.sum(1).tolist(),common_rms_a=float(ba[both].mean()) if both.any() else None,
                common_rms_b=float(bb[both].mean()) if both.any() else None)

def summary(panel):
    if panel.startswith("arrangement"):
        registry=json.loads((V3/"geometry/audits/arrangement-frozen.json").read_text())
        ids=registry["covered_ids" if panel.endswith("control") else "uncovered_ids"]
    else:ids=SPLITS[panel+"_ids"]
    sp="test" if panel=="test" else "train" if panel in ("family","shared") else "val"
    with np.load(ROOT/f"scaled/mp20-data/{sp}.npz") as ref:
        look={str(v):i for i,v in enumerate(ref["ids"])};z=ref["z"][[look[x] for x in ids]]
    with np.load(ROOT/"scaled/mp20-data/train.npz") as raw:
        use=np.isin(raw["ids"],SPLITS["train_ids"])
        chem={tuple(np.sort(row[row>0])) for row in raw["z"][use]}
    n=(z>0).sum(1)
    seen=np.array([tuple(np.sort(row[row>0])) in chem for row in z])
    strata=dict(all=np.ones(len(ids),bool),large=n>=9,seen=seen,unseen=~seen)
    duplicates=HERE/"duplicates.csv"
    if panel=="test" and duplicates.is_file():
        with duplicates.open() as fp:
            exclude={r["id"] for r in csv.DictReader(fp) if r["split"]=="test" and r["earlier_training_match"]}
        strata["no_tight_training_match"]=~np.isin(ids,list(exclude))
    strata.update({f"{lo}-{hi}":(n>=lo)&(n<=hi) for lo,hi in ((1,4),(5,8),(9,12),(13,20))})
    source={}
    for file in sorted((HERE/"scores"/panel).glob("*/outcomes.npz")):
        source[file.parent.name]=[file]
    for file in sorted((V3/"uflow/scores").glob("*/frozen/"+panel+"/outcomes.npz")):
        source["uflow-"+file.parents[2].name]=[file]
    if panel=="test":
        v2=V3.parent/"v2/csp/test"
        source.update({"released-angular":[v2/f"angular-seed{s}/outcomes.npz" for s in (11,23,37)],
                       "matched16":[v2/f"flow16-seed{s}/outcomes.npz" for s in (11,23,37)],
                       "released-uflow":[ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz"]})
    # The same zero-call stream is shared across fits. Average training seeds at fixed target.
    for tag in list(source):
        if tag.endswith("-11"):
            base=tag[:-3]
            if all(base+"-"+str(x) in source for x in (11,23,37)):
                source[base+"-3seeds"]=[source[base+"-"+str(x)][0] for x in (11,23,37)]
    data={m:[load(p,ids) for p in paths] for m,paths in source.items()}
    quality={};diff={}
    for strat,sl in strata.items():
        if not sl.any():continue
        cluster=None
        if panel in ("family","shared"):
            cluster=np.array([SPLITS["family_by_challenge_id"][x] for x in ids])[sl]
        elif panel.startswith("arrangement"):
            cluster=np.array([str(registry["key_by_id"][x]) for x in ids])[sl]
        for k in (1,5,20):
            cache={}
            for name,sets in data.items():
                if sets[0]["r"].shape[1]<k:continue
                q,c=metrics(sets,sl,k);quality[f"{strat}/{k}/{name}"]=q;cache[name]=c
            for a in cache:
                baselines=["prior","released-uflow","geometry-donor","uflow-prior-11"]
                baselines+=["angular-small-11","angular-small-3seeds","angular-medium-11","angular-medium-3seeds","angular-medium-long","meanflow-medium-long",
                            "uflow-medium-11-long","uflow-small-11-long"]
                for b in dict.fromkeys(baselines):
                    if a==b or b not in cache:continue
                    diff[f"{strat}/{k}/{a}--{b}"]=contrasts(cache[a],cache[b],cluster)
    out=HERE/"reports";out.mkdir(exist_ok=True)
    (out/(panel+".json")).write_text(json.dumps(dict(panel=panel,ids=ids,source={m:[p.relative_to(ROOT).as_posix() for p in paths] for m,paths in source.items()},
        quality=quality,paired=diff),indent=1,allow_nan=False))
    print(panel,len(ids),len(source),"methods",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("panels",nargs="+")
    for p in ap.parse_args().panels:summary(p)
