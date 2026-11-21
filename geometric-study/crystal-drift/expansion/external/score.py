"""Score an ordered CrystalFlow pool with the sealed MP-20 matcher, saving every outcome."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[var]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"revision/full"))
from match_full import candidate,target_context,unpack_gram,pin
from collections import Counter
def init():
    global REF
    pin()
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as x:REF={k:x[k] for k in ("ids","z","k","f")}

@lru_cache(maxsize=768)
def context_at(index):
    return target_context(REF["k"][index],REF["f"][index],REF["z"][index])
def score(path):
    out=path.parent/"scores"/path.name
    if out.exists():return
    with np.load(path) as x:
        ids,z,k,f,d,start=[x[key] for key in ("ids","z","k","f","draw","first")]
    end=start+len(ids); np.testing.assert_array_equal(ids,REF["ids"][start:end])
    np.testing.assert_array_equal(z,REF["z"][start:end])
    v=k[:,:3].sum(1)-REF["k"][start:end,:3].sum(1,dtype=float)
    rms=np.full(len(ids),np.nan);fit=np.zeros(len(ids),bool);errors=Counter();checks=0
    for j,i in enumerate(range(start,end)):
        try:
            ctx=context_at(i)
            verify=(i*104729+int(d)*1543)%1031==0;checks+=int(verify)
            gg=unpack_gram(k[j:j+1])[0]
            rms[j],fit[j],_=candidate(ctx,k[j],f[j],z[j][z[j]>0].astype(np.int64),gg,verify)
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as exc:
            errors[type(exc).__name__]+=1
    temp=out.with_suffix(".part.npz")
    np.savez_compressed(temp,ids=ids,draw=d,first=start,rms=rms,strict_fit=fit,log_volume_ratio=v,
            exceptions=json.dumps(errors),public_api_checks=checks)
    temp.replace(out)

def report(folder):
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as x:ids=x["ids"]
    shape=(len(ids),20);rms=np.full(shape,np.nan);strict=np.zeros(shape,bool);lv=np.full(shape,np.nan);seen=np.zeros(shape,bool)
    for p in sorted((folder/"scores").glob("draw*.npz")):
        if ".part." in p.name:continue
        with np.load(p) as x:
            a=int(x["first"]);b=a+len(x["ids"]);j=int(x["draw"])
            np.testing.assert_array_equal(ids[a:b],x["ids"]);assert not seen[a:b,j].any();seen[a:b,j]=True
            for target,key in ((rms,"rms"),(strict,"strict_fit"),(lv,"log_volume_ratio")):target[a:b,j]=x[key]
    summary=folder/"summary.json"
    rows=json.loads(summary.read_text()) if summary.exists() else {}
    for n in (128,len(ids)):
        for k in (1,5,20):
            if not seen[:n,:k].all():continue
            if f"{n}@{k}" in rows:continue
            a=rms[:n,:k];hit=np.isfinite(a);best=np.where(hit,a,np.inf).min(1);good=hit.any(1)
            vol=(hit & (np.abs(lv[:n,:k])<=np.log(1.1))).any(1)
            rows[f"{n}@{k}"]=dict(targets=n,k=k,hits=int(good.sum()),mr=good.mean(),rmse=best[good].mean(),
                maximum_site=strict[:n,:k].any(1).mean(),volume_qualified=vol.mean())
            archive=np.load(ROOT/"revision/full/mp20/complete-crosswalk-k20.npz");index=list(archive["arms"]).index("drift-seed11")
            drift=np.isfinite(archive["rms"][:n,index:index+3,:k]).any(2).mean(1)
            gap=good-drift
            rng=np.random.default_rng(48180)
            sample=np.concatenate([gap[rng.integers(0,n,(100,n))].mean(1) for i in range(100)])
            rows[f"{n}@{k}"]["minus_frozen_drift"]=[float(gap.mean()),*np.quantile(sample,[.025,.975]).tolist()]
            archive.close()
    np.savez_compressed(folder/"score-crosswalk.npz",ids=ids,rms=rms,strict_fit=strict,log_volume_ratio=lv,scored=seen)
    (folder/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    print("scored",seen.sum(),"candidates",list(rows),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("pool",type=Path);ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--watch",action="store_true");ap.add_argument("--target-tiles",type=int,default=720);a=ap.parse_args()
    dst=a.pool/"scores";dst.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers,initializer=init) as pool:
        while True:
            todo=[p for p in sorted(a.pool.glob("draw*.npz")) if ".part." not in p.name and not (dst/p.name).exists()]
            if todo:
                # Group available draws from each target tile to reuse its reduced structures.
                todo.sort(key=lambda p:(int(p.name[4:6])>=5,p.name[7:],p.name[:6]))
                list(pool.map(score,todo));report(a.pool)
            completed=sum(".part." not in p.name for p in dst.glob("draw*-cases*.npz"))
            if not a.watch or completed>=a.target_tiles:break
            time.sleep(15)
