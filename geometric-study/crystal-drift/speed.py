"""Interleaved wall-clock measurements from supplied composition to a materialized cell."""
import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

from baseline import Velocity
from baseline_multi import MultiVelocity
from model import CrystalGenerator


HERE = Path(__file__).resolve().parent
SAVE = HERE/"speed-results"
SEEDS = [11,23,37]


def checkpoint_path(cohort, method, seed):
    if method == "drift":
        return HERE/"drift-results"/(cohort+"-exact")/f"seed{seed}.pt"
    folder = "baseline-results" if cohort=="ab" else "baseline-multi-results"
    return HERE/folder/f"seed{seed}.pt"


def portable_path(cohort, method, seed):
    return SAVE/"portable"/f"{cohort}-{method}-{seed}.npz"


def export():
    (SAVE/"portable").mkdir(parents=True, exist_ok=True)
    for cohort in ("ab","multi"):
        for method in ("drift","flow"):
            for seed in SEEDS:
                ck = torch.load(checkpoint_path(cohort,method,seed),map_location="cpu",weights_only=True)
                values = {k:v.numpy() for k,v in ck["model"].items()}
                np.savez_compressed(portable_path(cohort,method,seed),mean=ck["mean"].numpy(),
                                    scale=ck["scale"].numpy(),**values)


def load_model(cohort, method, seed, device):
    model = CrystalGenerator() if method=="drift" else (Velocity() if cohort=="ab" else MultiVelocity())
    with np.load(portable_path(cohort,method,seed)) as file:
        model.load_state_dict({k:torch.from_numpy(file[k].copy()) for k in model.state_dict()})
        mean, scale = [torch.from_numpy(file[k].copy()).to(device) for k in ("mean","scale")]
    return model.eval().to(device), mean, scale


def physical_cell(k):
    s = np.zeros((len(k),3,3),dtype=np.float64)
    for j in range(3):
        s[:,j,j] = k[:,j]
    for j,(a,b) in enumerate(((0,1),(0,2),(1,2))):
        s[:,a,b] = s[:,b,a] = k[:,j+3]/math.sqrt(2)
    w,q = np.linalg.eigh(s)
    return (q * np.exp(w)[:,None,:]) @ q.transpose(0,2,1)


def donors(train_z, test_z):
    # A warm retrieval service can cache the donor pool for each requested composition.
    # Construction is reported separately from the timed lookup, draw and cell decoding.
    cache = {}
    zlohi = np.stack((train_z[:,0],train_z.max(1)),axis=1)
    ns = (train_z>0).sum(1)
    low = (train_z==train_z[:,:1]).sum(1)
    for row in np.unique(test_z,axis=0):
        ix = np.flatnonzero(np.all(train_z==row,axis=1))
        if not len(ix):
            n, a = (row>0).sum(), (row==row[0]).sum()
            pool = np.flatnonzero((ns==n)&(low==a))
            if not len(pool):
                pool = np.flatnonzero(ns==n)
            d = np.abs(zlohi[pool]-np.array([row[0],row.max()])).sum(1)
            ix = pool[d==d.min()]
        cache[tuple(row)] = ix
    return cache


@torch.inference_mode()
def draw(model_info, cohort, method, atoms, device):
    model, mean, scale = model_info
    z = torch.from_numpy(atoms).to(device=device,dtype=torch.long)
    k = torch.randn(len(z),6,device=device)
    if cohort=="ab":
        f = torch.rand(len(z),3,device=device)
    else:
        f = torch.rand(len(z),z.shape[1],3,device=device)
    if method=="drift":
        if cohort=="ab":
            f = torch.stack((torch.zeros_like(f), f),dim=1)
        k,f = model(z,k,f)
    else:
        steps = int(method[4:])
        if cohort=="multi":
            condition = model.prepare(z)
            mask = condition[0]
            f = f*mask
        for j in range(steps):
            t = torch.full((len(z),1),j/steps,dtype=k.dtype,device=device)
            if cohort=="ab":
                v = model(z,k,f,t)
                dk,df = v[:,:6],v[:,6:]
            else:
                dk,df = model(condition,k,f,t)
            k = k + dk/steps
            f = (f + df/(math.sqrt(12)*steps)).remainder(1)
            if cohort=="multi":
                f = f*mask
        if cohort=="ab":
            f = torch.stack((torch.zeros_like(f), f),dim=1)
    cell = physical_cell((k * scale + mean).cpu().numpy())
    positions = f.cpu().numpy()
    return cell, positions


def measure(device, repeats):
    torch.set_num_threads(1)
    torch.manual_seed(17566)
    rng = np.random.default_rng(17566)
    rows = []
    setup = {}
    for cohort, prefix in (("ab",""),("multi","multi-")):
        with np.load(HERE/"data"/(prefix+"test.npz")) as d:
            test = d["z"]
        with np.load(HERE/"data"/(prefix+"train.npz")) as d:
            train = {k:d[k] for k in ("z","k","f")}
        tic = time.perf_counter()
        cache = donors(train["z"],test)
        setup[cohort] = time.perf_counter()-tic
        methods = ["drift","flow1","flow16","retrieval"]
        if cohort=="ab":
            methods.insert(2,"flow4")
        for seed in SEEDS:
            models = {key:load_model(cohort,key,seed,device) for key in ("drift","flow")}
            for batch in (1,64,256):
                for rep in range(-3,repeats):
                    atoms = test[rng.integers(len(test),size=batch)]
                    for method in rng.permutation(methods):
                        if device=="cuda":
                            torch.cuda.synchronize()
                        tic = time.perf_counter_ns()
                        if method=="retrieval":
                            pools = [cache[tuple(z)] for z in atoms]
                            indices = [p[int(u*len(p))] for p,u in zip(pools,rng.random(len(pools)))]
                            cell = physical_cell(train["k"][indices])
                            f = train["f"][indices]
                            if cohort=="ab":
                                f = np.stack((np.zeros_like(f),f),axis=1)
                        else:
                            key = "drift" if method=="drift" else "flow"
                            cell,f = draw(models[key],cohort,method,atoms,device)
                        if device=="cuda":
                            torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns()-tic)*1e-6
                        assert np.isfinite(cell).all() and np.isfinite(f).all()
                        if rep >= 0:
                            rows.append(dict(cohort=cohort,device=device,method=method,seed=seed,
                                             batch=batch,repeat=rep,wall_ms=elapsed))
    fields = list(rows[0])
    with (SAVE/f"timings-{device}.csv").open("w",newline="",encoding="utf8") as file:
        writer = csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
                    platform=platform.platform(),processor=platform.processor(),threads=torch.get_num_threads(),
                    accelerator=torch.cuda.get_device_name() if device=="cuda" else "CPU",cuda=torch.version.cuda,
                    repeats=repeats,untimed_warmups=3,warm_donor_cache_build_seconds=setup)
    (SAVE/f"timings-{device}.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf8")
    print(device,len(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export",action="store_true")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--repeats",type=int,default=15)
    args=p.parse_args()
    if args.export:
        export()
    measure(args.device,args.repeats)
