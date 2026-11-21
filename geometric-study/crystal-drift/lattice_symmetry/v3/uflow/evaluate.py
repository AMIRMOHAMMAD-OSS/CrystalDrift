"""Use the shared pinned comparator, lock half/full checkpoints, and retain every scored attempt."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"evaluation"))
from score import score
import numpy as np

def one(job):
    source,split,dest=job
    score(source,split,dest)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(tag,mode,workers,panels):
    root=HERE/"predictions"/tag
    folders=sorted(root.glob("ema-*/dev/ready.json")) if mode=="dev" else [
        root/"frozen"/panel/"ready.json" for panel in panels]
    if not folders:
        raise ValueError("No generated development pools to score")
    results={}
    summaries={}
    for ready in folders:
        receipt=json.loads(ready.read_text())
        directory=ready.parent
        target=HERE/"scores"/tag/directory.parent.name/directory.name
        target.mkdir(exist_ok=True,parents=True)
        jobs=[(p,receipt["split"],target/p.name) for p in sorted(directory.glob("*.npz"))]
        if workers>1 and len(jobs)>1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one,jobs))
        else:
            for j in jobs:
                one(j)
        rows=[]
        for p,split,d in jobs:
            with np.load(d) as data:
                assert str(data["source_sha"])==digest(p)
                rows.append({k:data[k] for k in ("ids","rms","strict_fit","log_volume_ratio","error")})
        out={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        assert len(out["ids"])==len(set(out["ids"]))==receipt["count"]
        target_file=target/"outcomes.npz"
        np.savez_compressed(target_file,**out)
        result={}
        for k in [n for n in (1,5,20) if n<=receipt["draws"]]:
            rms=out["rms"][:,:k]
            ok=np.isfinite(rms)
            best=np.where(ok,rms,np.inf).min(1)
            native=ok & (np.abs(out["log_volume_ratio"][:,:k])<=np.log(1.1))
            result[k]=dict(hits=int(ok.any(1).sum()),targets=len(rms),mr=float(ok.any(1).mean()),
                mean_best_rms=float(best[np.isfinite(best)].mean()) if np.isfinite(best).any() else None,
                native_hits=int(native.any(1).sum()),strict_hits=int(out["strict_fit"][:,:k].any(1).sum()),
                errors=int(out["error"][:,:k].astype(bool).sum()))
        summaries[str(target.relative_to(HERE))]=result
        (target/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
        results[directory.parent.name]=result[5]["hits"]
    if mode=="dev":
        assert len(results)==2
        if tag.endswith("-long"):
            original_fit=HERE/"fits"/tag[:-5]
            early=json.loads((original_fit/"selection.json").read_text())["results"]
            results.update(early)
        winner=min(results,key=lambda n:(-results[n],n))
        fit=HERE/"fits"/tag
        ck=fit/(winner+".pt")
        if not ck.exists() and tag.endswith("-long"):
            ck=fit.parent/tag[:-5]/(winner+".pt")
        import os
        freeze=dict(cohort="dev",results=results,rule="earliest maximum five-draw recovery",
                    checkpoint=os.path.relpath(ck,fit).replace("\\","/"),checkpoint_sha=digest(ck),split_sha=digest(HERE.parent/"protocol/splits.json"))
        out=fit/"selection.json"
        if out.exists():
            assert freeze==json.loads(out.read_text())
        out.write_text(json.dumps(freeze,indent=2)+"\n")
        print(tag,freeze,flush=True)
    print(json.dumps(summaries),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("mode",choices=("dev","frozen"))
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--panels",nargs="+",default=["confirm","family","shared","test"])
    a=p.parse_args()
    evaluate(a.tag,a.mode,a.workers,a.panels)
