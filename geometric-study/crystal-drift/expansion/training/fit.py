"""Train one uninterrupted trajectory per method and seed with paired composition schedules."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"

import numpy as np
import torch
import psutil

from controls import ROOT, build, meanflow_loss
from experiment import load, group_indices
from family import objective

HERE=Path(__file__).resolve().parent
METHODS=("drift","iid","flow","meanflow")
PASSES=(1,3,10)


def write(path,data):
    tmp=path.with_suffix(".writing")
    tmp.write_text(json.dumps(data,indent=1)+"\n",encoding="utf8")
    tmp.replace(path)


def fit(job):
    method,seed,cores=job
    if cores and hasattr(psutil.Process(),"cpu_affinity"):
        psutil.Process().cpu_affinity(cores)
    torch.set_num_threads(1)
    work=HERE/"fits"/f"{method}-seed{seed}"
    work.mkdir(exist_ok=True,parents=True)
    if (work/"pass10.json").is_file():
        return
    data=load(ROOT/"scaled/mp20-data/train.npz")
    z=torch.from_numpy(data["z"]).long()
    f=torch.from_numpy(data["f"]).float()
    k=torch.from_numpy(data["k"]).float()
    mu=k.mean(0)
    sd=k.std(0,unbiased=False).clamp_min(1e-5)
    k=(k-mu)/sd
    labels,table,starts,counts=group_indices(data["z"])
    rng=np.random.default_rng(seed+321)
    torch.manual_seed(seed)
    net=build(method)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    positive_seen=np.zeros(len(k),bool)
    anchors_seen=np.zeros(len(k),bool)
    perm=np.array([],dtype=int)
    schedule=hashlib.sha256()
    bankhash=hashlib.sha256()
    tic=perf_counter()
    cpu0=psutil.Process().cpu_times()
    history=[]
    steps=len(z)//16
    assert steps*16==len(z)
    for i in range(steps*max(PASSES)):
        if len(perm)<16:
            perm=np.concatenate((perm,rng.permutation(len(z))))
        anchors,perm=perm[:16],perm[16:]
        gids=labels[anchors]
        picks=(rng.random((16,4))*counts[gids,None]).astype(int)
        indices=table[starts[gids,None]+picks]
        if method!="iid":
            indices[:,0]=anchors
        schedule.update(anchors.astype("<i4").tobytes())
        bankhash.update(indices.astype("<i4").tobytes())
        positive_seen[indices]=True
        anchors_seen[anchors]=True
        ix=torch.from_numpy(indices.ravel())
        if method=="meanflow":
            loss=meanflow_loss(net,z[ix],k[ix],f[ix])
        else:
            loss=objective(net,"flow" if method=="flow" else "drift",z[ix],k[ix],f[ix],4)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        opt.step()
        step=i+1
        if step%200==0 or step%steps==0:
            if not torch.isfinite(norm):
                raise FloatingPointError(f"{method} {seed} {step}")
            history.append([step,float(loss.detach()),float(norm)])
        if step%steps or step//steps not in PASSES:
            continue
        e=step//steps
        path=work/f"pass{e}.pt"
        tmp=path.with_suffix(".building")
        torch.save(dict(model=net.state_dict(),mean=mu,scale=sd,method=method,seed=seed,passes=e),tmp)
        tmp.replace(path)
        cput=psutil.Process().cpu_times()
        info=dict(method=method,seed=seed,passes=e,updates=step,batch=64,group_size=4,
            positive_examples=step*64,query_examples=step*64,
            generated_reference_examples=step*64 if method in ("drift","iid") else 0,
            unique_positive_rows=int(positive_seen.sum()),unique_schedule_anchors=int(anchors_seen.sum()),
            schedule_sha256=schedule.hexdigest(),bank_sha256=bankhash.hexdigest(),parameters=sum(p.numel() for p in net.parameters()),
            elapsed_seconds=perf_counter()-tic,process_cpu_seconds=cput.user+cput.system-cpu0.user-cpu0.system,
            gpu_hours=0,platform=platform.platform(),torch=torch.__version__,cpu_threads=1,
            affinity=cores,loss_trace=history,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write(work/f"pass{e}.json",info)
        print(method,seed,e,round(info["elapsed_seconds"],2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    parser.add_argument("--seeds",nargs="+",type=int,default=(11,23,37))
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--cores",nargs="+",type=int,default=None)
    args=parser.parse_args()
    jobs=[(m,s,args.cores) for s in args.seeds for m in args.methods]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fit,jobs))
