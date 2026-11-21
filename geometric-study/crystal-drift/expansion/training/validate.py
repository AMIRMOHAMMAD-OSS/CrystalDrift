"""Score all available checkpoints on the fixed validation cohort and freeze selections."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"

import numpy as np
import torch
from controls import ROOT,build,generate
from experiment import load,select
from evaluate import process
from fit import HERE, METHODS, PASSES, write


def save_npz(path,**arrays):
    tmp=path.with_suffix(".building")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream,**arrays)
    tmp.replace(path)


@torch.inference_mode()
def predict(state,method,z,k,f,steps,batch=192):
    net=build(method).eval()
    net.load_state_dict(state["model"])
    outk=[]; outf=[]
    for i in range(0,len(z),batch):
        s=slice(i,i+batch)
        a,b=generate(net,method,torch.from_numpy(z[s]).long(),torch.from_numpy(k[s]),torch.from_numpy(f[s]),steps)
        outk.append((a*state["scale"]+state["mean"]).numpy())
        outf.append(b.numpy())
    return np.concatenate(outk),np.concatenate(outf)


def validation(checkpoint,data,physical,scales,steps):
    state=torch.load(checkpoint,weights_only=True,map_location="cpu")
    method=state["method"];seed=state["seed"];ndraw=5
    z=np.repeat(data["z"],ndraw,axis=0)
    rng=np.random.default_rng(seed+100000)
    k0=rng.standard_normal((len(z),6)).astype(np.float32)
    f0=rng.random((len(z),20,3)).astype(np.float32)
    kg,fg=predict(state,method,z,k0,f0,steps)
    g=process(kg,fg,z).reshape(-1,ndraw,4)
    assert np.isfinite(g).all()
    a=g/scales
    obs=physical/scales
    es=np.linalg.norm(a-obs[:,None],axis=-1).mean(1)-np.linalg.norm(a[:,:,None]-a[:,None,:],axis=-1).sum((1,2))/(2*ndraw*(ndraw-1))
    k=kg.reshape(-1,ndraw,6);f=fg.reshape(-1,ndraw,20,3)
    suffix=f"{checkpoint.stem}-val-{steps}"
    save_npz(checkpoint.parent/(suffix+".npz"),ids=data["ids"],z=data["z"],k=k,f=f,geometry=g,target_geometry=physical,energy=es)
    write(checkpoint.parent/(suffix+".json"),dict(passes=state["passes"],method=method,seed=seed,nfe=steps,energy=float(es.mean()),n_ids=len(data["ids"]),n_draws=ndraw))


def selections():
    out={}
    rows=[]
    for method in METHODS:
        step=16 if method=="flow" else 1
        energies={}
        for epoch in PASSES:
            scores=[]
            for seed in (11,23,37):
                path=HERE/f"fits/{method}-seed{seed}/pass{epoch}-val-{step}.json"
                if path.exists():
                    rec=json.loads(path.read_text());scores.append(rec["energy"]);rows.append(rec)
            if len(scores)==3:energies[epoch]=float(np.mean(scores))
        if len(energies)==len(PASSES):
            out[method]={"pass":min(energies,key=energies.get),"validation_energy":energies,"validation_ids":128,"draws_per_id":5}
    write(HERE/"selection.json",out)
    if rows:
        with (HERE/"validation-curves.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(out)==len(METHODS)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch",action="store_true");args=ap.parse_args()
    torch.set_num_threads(1)
    val=load(ROOT/"scaled/mp20-data/val.npz")
    idx=select(val["ids"],128)
    subset={k:v[idx] for k,v in val.items()}
    physical=process(subset["k"],subset["f"],subset["z"])
    scales=np.array(json.loads((ROOT/"scaled/runs/mp20/score-scale.json").read_text())["scale"])
    while True:
        for run in sorted((HERE/"fits").glob("*-seed*")):
            m=run.name.split("-seed")[0]
            for epoch in PASSES:
                ck=run/f"pass{epoch}.pt"
                if not (run/f"pass{epoch}.json").exists():continue
                for steps in ((1,16) if m=="flow" else (1,)):
                    if not (run/f"pass{epoch}-val-{steps}.json").exists():
                        validation(ck,subset,physical,scales,steps)
                        print(run.name,epoch,steps,flush=True)
        if selections() or not args.watch:break
        time.sleep(4)


if __name__=="__main__":main()
