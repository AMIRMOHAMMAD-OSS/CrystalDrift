"""Sample 10,000 common empirical chemistries, fresh orbit coordinates and matched neural updates."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.models.sampler import Sampler
from lattice_symmetry.v3.evaluation.predict import atomic_npz,h
from data import pack_gram

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("arm");ap.add_argument("--checkpoint",type=Path);ap.add_argument("--released",action="store_true")
    ap.add_argument("--batch",type=int,default=32);a=ap.parse_args()
    torch.set_num_threads(1);s=Sampler();w=s.load(a.checkpoint) if a.checkpoint else None
    if a.released:
        from lattice_symmetry.v2.csp.model import load_service
        v2=load_service("angular",11)
    keep=set(json.loads((V3/"protocol/splits.json").read_text())["train_ids"])
    with np.load(ROOT/"scaled/mp20-data/train.npz") as p:
        train=p["z"][np.isin(p["ids"],list(keep))];tids=p["ids"][np.isin(p["ids"],list(keep))]
    pool=HERE/"pools";pool.mkdir(exist_ok=True)
    zarr=np.zeros((10000,20),np.int16);farr=np.zeros((10000,20,3),np.float32)
    karr=np.full((10000,6),np.nan);good=np.zeros(10000,bool);g=np.zeros(10000,np.int16)
    donors=np.full(10000,"",dtype="U22");cells=donors.copy();form_id=donors.copy();patterns=[]
    rng=np.random.default_rng(60611);t=perf_counter()
    for i in range(0,10000,a.batch):
        idx=rng.integers(len(train),size=min(10000-i,a.batch))
        zz=train[idx];form_id[i:i+len(idx)]=tids[idx]
        if a.released:
            from data import unpack_gram
            kk,ff=v2(zz,np.random.default_rng(np.random.SeedSequence([869332,i])))
            outs=[(np.linalg.cholesky(unpack_gram(ki)),fi[zi>0],zi[zi>0]) for ki,fi,zi in zip(kk,ff,zz)]
            patterns.extend(["[]"]*len(zz))
        else:
            seeds=[]
            for j,z in enumerate(zz,i):
                seed=s.seeds(z,1,np.random.SeedSequence([869332,j]))[0]
                seeds.append(seed);donors[j]=seed.donor;cells[j]=seed.cell_donor;g[j]=seed.group
                patterns.append(json.dumps([(int(z0),l) for z0,l in seed.template],separators=(",",":")))
            outs=s.apply(w,seeds)
        for j,(L,F,Z) in enumerate(outs,i):
            zarr[j,:len(Z)]=Z;farr[j,:len(Z)]=F
            if np.isfinite(L).all() and np.isfinite(F).all() and abs(np.linalg.det(L))>.1:
                karr[j]=pack_gram(L@L.T);good[j]=True
    elapsed=perf_counter()-t
    path=pool/(a.arm+".npz")
    atomic_npz(path,ids=[f"v3-{i+1:05}" for i in range(10000)],z=zarr,k=karr,f=farr,valid_output=good,
               group=g,donor=donors,cell_donor=cells,chemistry_source=form_id,patterns=patterns)
    (pool/(a.arm+".json")).write_text(json.dumps(dict(draws=10000,accepted=int(good.sum()),seconds=elapsed,
         cpu_threads=1,batch=a.batch,sha=h(path),ckpt=h(a.checkpoint) if a.checkpoint else ("v2 angular-11" if a.released else "prior"))))
    print(a.arm,good.sum(),elapsed,flush=True)

if __name__=="__main__":main()
