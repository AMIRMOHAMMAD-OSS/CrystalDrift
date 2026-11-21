"""Generate and score ordered draws given only element-labelled occupied template and Hall setting."""
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import numpy as np
import torch
from charts import HERE
from geometry import cohort,quality,reference
from labels import ROOT
from prior import Prior,records
from train_template import apply,draw,signature
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v2.physical import Generator
from lattice_symmetry.v2.csp.model import checkpoint_path

def model_and_hash(arm):
    if arm=="prior":return None,dict(method="train-only conditional metric and random affine orbit coordinates",nfe=0)
    if arm=="angular-A":
        path=checkpoint_path("angular",11)
    else:
        folder={"angular-C":"training","orbit-C":"training-orbit"}[arm]
        choice=json.loads((HERE/folder/"selection.json").read_text())
        path=HERE/folder/choice["chosen"]
        assert choice["chosen_sha256"]==signature(path)
    state=torch.load(path,map_location="cpu",weights_only=True)
    net=Generator().eval()
    net.load_state_dict(state["model"],strict=True)
    return net,dict(checkpoint=str(path.relative_to(ROOT)),sha256=signature(path),nfe=1)

def generate(arm,k=20):
    targets=cohort("test")
    if k!=20:raise ValueError("Keep 20 draws per target")
    net,meta=model_and_hash(arm);prior=Prior(records())
    rng=np.random.default_rng(203701);torch.set_num_threads(1)
    N=max(t.get("std_n",0) for t in targets)
    l=np.full((128,k,3,3),np.nan);f=np.full((128,k,N,3),np.nan);z=np.zeros((128,N),np.int16)
    seconds=[];errors={}
    for i,r in enumerate(targets):
        start=perf_counter()
        try:
            if "error" in r:raise ValueError(r["error"])
            proposals=[draw(r,prior,rng) for _ in range(k)]
            if net is not None:proposals=apply(net,proposals)
            for j,(L,F,Z) in enumerate(proposals):
                l[i,j]=L;f[i,j,:len(Z)]=F;z[i,:len(Z)]=Z
        except Exception as exc:errors[i]=repr(exc)
        seconds.append(perf_counter()-start)
    dest=HERE/"pools";dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f"{arm}.npz",id=[r["id"] for r in targets],L=l,F=f,Z=z,elapsed=seconds)
    meta.update(draws=k,count=128,errors=errors,seed=203701,cpu_threads=1,
                seconds=sum(seconds),torch=torch.__version__,pool_sha256=signature(dest/f"{arm}.npz"))
    (dest/f"{arm}.json").write_text(json.dumps(meta,indent=2)+"\n")

def score(arm):
    dest=HERE/"pools";targets=cohort("test")
    with np.load(dest/f"{arm}.npz") as b:l,f,z,ids=b["L"],b["F"],b["Z"],b["id"]
    assert ids.tolist()==[r["id"] for r in targets]
    result=HERE/"scores";result.mkdir(exist_ok=True)
    temp=result/f"{arm}.part"
    with gzip.open(temp,"wt",encoding="utf8") as fh:
        for i,r in enumerate(targets):
            ref=reference(r,"test")
            for j in range(f.shape[1]):
                v=dict(arm=arm,id=r["id"],index=i,draw=j+1)
                try:
                    mask=z[i]>0
                    if not np.isfinite(l[i,j]).all():raise ValueError("unavailable draw")
                    v.update(quality(l[i,j],f[i,j,mask],z[i,mask],r,ref))
                except Exception as exc:v["error"]=repr(exc)
                fh.write(json.dumps(v,separators=(",",":"))+"\n")
    temp.replace(result/f"{arm}.jsonl.gz")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm");p.add_argument("--score-only",action="store_true");a=p.parse_args()
    if not a.score_only:generate(a.arm)
    score(a.arm)
