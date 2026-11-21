"""Train-only prediction of native volume, isotropic reconstruction and rematching."""
import argparse
import json
import os
import sys
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[key]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"revision/full"),str(ROOT)]
from match_full import candidate,target_context
from data import unpack_gram
BASE=ROOT/"revision/full/mp20"
SEEDS=(11,23,37)


def composition(row):
    elements,counts=np.unique(row[row>0],return_counts=True)
    return tuple(zip(elements.tolist(),(counts//np.gcd.reduce(counts)).tolist()))


def covariates(z):
    x=np.zeros((len(z),119),float)
    for j,row in enumerate(z):
        counts=np.bincount(row,minlength=119)
        x[j]=counts/max(1,(row>0).sum())
        x[j,0]=1.
    return x


def train_and_correct():
    with np.load(ROOT/"scaled/mp20-data/train.npz") as source:
        k,z,ids=(source[n] for n in ("k","z","ids"))
    x=covariates(z);y=k[:,:3].sum(1,dtype=float)-np.log((z>0).sum(1))
    penalty=np.eye(x.shape[1]);penalty[0,0]=0
    weights=np.linalg.solve(x.T@x+penalty,x.T@y)
    # Smoothed exact-formula residual on the same training data.
    residual={}
    for row,err in zip(z,y-x@weights):residual.setdefault(composition(row),[]).append(err)
    with np.load(BASE/"cohort.npz") as file:
        zt,targets=file["z"],file["ids"]
    assert set(targets).isdisjoint(ids)
    pred=covariates(zt)@weights+np.log((zt>0).sum(1))
    for i,row in enumerate(zt):
        empirical=residual.get(composition(row),[])
        pred[i]+=sum(empirical)/(3+len(empirical))
    np.savez_compressed(HERE/"model.npz",weights=weights,train_ids=ids,ids=targets,predicted_log_cell_volume=pred)
    for seed in SEEDS:
        original=np.load(BASE/f"drift-seed{seed}/k.npy",mmap_mode="r")
        new=np.array(original,dtype=np.float64)
        new[:,:,:3]+=(pred[:,None]-new[:,:,:3].sum(2))[:,:,None]/3
        np.testing.assert_allclose(new[:,:,:3].sum(2),np.broadcast_to(pred[:,None],new.shape[:2]),atol=1e-12)
        np.save(HERE/f"corrected-seed{seed}.npy",new)


@lru_cache(maxsize=1)
def resources():
    with np.load(BASE/"cohort.npz") as f:refs={k:f[k] for k in ("ids","z","k","f")}
    preds=[(np.load(HERE/f"corrected-seed{s}.npy",mmap_mode="r"),np.load(BASE/f"drift-seed{s}/f.npy",mmap_mode="r")) for s in SEEDS]
    return refs,preds


def match(lo):
    path=HERE/"tiles"/f"case-{lo:05d}.npz"
    if path.exists():return
    ref,preds=resources();hi=min(lo+32,len(ref["ids"]));shape=(hi-lo,len(SEEDS),20)
    rms=np.full(shape,np.nan);fit=np.zeros(shape,bool);vol=np.empty(shape);errors=0
    for i in range(lo,hi):
        ctx=target_context(ref["k"][i],ref["f"][i],ref["z"][i])
        live=ref["z"][i];live=live[live>0].astype(np.int64)
        for a,(ks,fs) in enumerate(preds):
            grams=unpack_gram(ks[i])
            for b,(k,f,g) in enumerate(zip(ks[i],fs[i],grams)):
                vol[i-lo,a,b]=k[:3].sum()-ref["k"][i,:3].sum(dtype=float)
                try:rms[i-lo,a,b],fit[i-lo,a,b],_=candidate(ctx,k,f,live,g,i%257==0 and b==0)
                except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
                    errors+=1
    with path.with_suffix(".part").open("wb") as fp:np.savez_compressed(fp,ids=ref["ids"][lo:hi],rms=rms,fit=fit,logvol=vol,errors=errors)
    path.with_suffix(".part").replace(path)


def summarize():
    tiles=[np.load(p) for p in sorted((HERE/"tiles").glob("*.npz"))]
    ids=np.concatenate([p["ids"] for p in tiles]);assert len(ids)==9046
    rms=np.concatenate([p["rms"] for p in tiles]);vol=np.concatenate([p["logvol"] for p in tiles])
    fit=np.concatenate([p["fit"] for p in tiles])
    with np.load(BASE/"complete-crosswalk-k20.npz") as prior:
        np.testing.assert_array_equal(ids,prior["ids"])
        ix=[prior["arms"].tolist().index(f"drift-seed{s}") for s in SEEDS]
        oldr=prior["rms"][:,ix];oldv=prior["log_volume_ratio"][:,ix]
        oldfit=prior["strict_fit"][:,ix]
    np.savez_compressed(HERE/"corrected-scores.npz",ids=ids,seeds=SEEDS,rms=rms,strict_fit=fit,log_volume_ratio=vol)
    rng=np.random.default_rng(82848);report={}
    for k in (1,5,20):
        for key,before,after in (("mr",np.isfinite(oldr),np.isfinite(rms)),
            ("volume_match",np.isfinite(oldr)&(np.abs(oldv)<=np.log(1.1)),np.isfinite(rms)&(np.abs(vol)<=np.log(1.1))),
            ("maximum_site",oldfit,fit)):
            a,b=[v[:,:,:k].any(2).mean(1) for v in (before,after)];delta=b-a
            distribution=np.concatenate([delta[rng.integers(0,len(ids),(80,len(ids)))].mean(1) for i in range(125)])
            report[f"{key}@{k}"]=dict(raw=float(a.mean()),corrected=float(b.mean()),difference=float(delta.mean()),ci=np.quantile(distribution,[.025,.975]).tolist())
    report["mean_absolute_log_volume_error"]=[float(np.abs(a).mean()) for a in (oldv,vol)]
    report["flipped_draws"]=int(np.logical_xor(np.isfinite(oldr),np.isfinite(rms)).sum())
    report["exceptions"]=sum(int(t["errors"]) for t in tiles)
    (HERE/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=2);a=ap.parse_args()
    (HERE/"tiles").mkdir(exist_ok=True)
    train_and_correct()
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        for i,_ in enumerate(executor.map(match,range(0,9046,32))):
            if i%40==0:print(i+1,"tiles",flush=True)
    summarize()
