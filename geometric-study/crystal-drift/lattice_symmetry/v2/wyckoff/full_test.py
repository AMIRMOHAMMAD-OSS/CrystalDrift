"""One ordered conditional draw for every official test ID, including rejected templates."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import torch
from charts import HERE
from evaluate import model_and_hash
from geometry import reference
from labels import read,SETTINGS,dataset
from prior import Prior,records
from train_template import apply,draw

def run():
    torch.set_num_threads(1);out=HERE/"full";out.mkdir(exist_ok=True)
    test=list(read("test"));ids=dataset("test")[0]
    assert len(test)==len(ids)==9046 and [r["id"] for r in test]==ids.tolist()
    prior=Prior(records());net,meta=model_and_hash("angular-C");matcher=StructureMatcher(**SETTINGS)
    rng=np.random.default_rng(205039);t=perf_counter()
    nmax=max(r["std_n"] for r in test if "error" not in r)
    cells=np.full((9046,2,3,3),np.nan); frac=np.full((9046,2,nmax,3),np.nan,dtype='f4')
    nums=np.zeros((9046,nmax),dtype='i2');errs=np.full((9046,2),np.nan)
    logs=[]
    for i,r in enumerate(test):
        rr=dict(id=r["id"],index=i)
        try:
            if "error" in r:raise ValueError(r["error"])
            seed=draw(r,prior,rng);pred=apply(net,[seed])[0]
            ref=reference(r,"test")
            for j,(L,F,Z) in enumerate([seed,pred]):
                cells[i,j]=L;frac[i,j,:len(Z)]=F;nums[i,:len(Z)]=Z
                match=matcher.get_rms_dist(ref,Structure(L,Z,F))
                if match is not None:errs[i,j]=float(match[0])
        except Exception as e:rr["error"]=repr(e)
        logs.append(rr)
        if i%2000==0:print(i,round(perf_counter()-t),flush=True)
    path=out/"one-draw.npz";np.savez_compressed(path,ids=ids,L=cells,F=frac,Z=nums,rms=errs,arms=["prior","angular-C"])
    with gzip.open(out/"counts.csv.gz","wt",newline="",encoding="utf8") as f:
        w=csv.writer(f);w.writerow(["id","index","error","prior_rms","angular_rms"])
        for row,pair in zip(logs,errs):w.writerow([row["id"],row["index"],row.get("error",""),*[v if np.isfinite(v) else "" for v in pair]])
    hit=np.isfinite(errs);delta=hit[:,1].astype(float)-hit[:,0];brng=np.random.default_rng(917157)
    samples=np.array([delta[brng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)])
    body=dict(cohort=9046,arms=["prior","angular-C"],seed=205039,checkpoint=meta,
        rejected=sum("error" in x for x in logs),hits=hit.sum(0).tolist(),mean_matched_rms=np.nanmean(errs,axis=0).tolist(),
        paired_angular_minus_prior=float(delta.mean()),paired_95=np.quantile(samples,[.025,.975]).tolist(),
        pool_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),wall_s=perf_counter()-t)
    (out/"summary.json").write_text(json.dumps(body,indent=2)+"\n")
    print(body,flush=True)
if __name__=="__main__":run()
