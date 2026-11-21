"""Resumable independent-bank MP-20 training with a common schedule across objectives."""
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time
import numpy as np
import psutil
import torch
from model import (ARMS,BANK,DESCRIPTOR,GENERATOR,GROUPS,HERE,ROOT,SEEDS,Prior,digest,flow_loss,
                   geometry,make_cell,new_model,save_npz,write_json)
from lattice_symmetry.v2.physical import blocks,features
from scaled.family import within_direction

MILESTONES=(16,64,256,1024,3392)


def train(job):
    arm,seed,stop,device=job
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    folder=HERE/"fits"/f"{arm}-{seed}"
    folder.mkdir(exist_ok=True,parents=True)
    trainpath=ROOT/"scaled/mp20-data/train.npz"
    with np.load(trainpath) as fp:z,k,f=[fp[v] for v in ("z","k","f")]
    _,gids,counts=np.unique(z,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(gids,kind="stable");starts=np.cumsum(counts)-counts
    schedule_rng=np.random.default_rng(seed+1501)
    anchors=schedule_rng.permutation(len(z)).reshape(-1,GROUPS)
    assert stop<=len(anchors)
    groups=gids[anchors]
    pick=(schedule_rng.random((*groups.shape,BANK))*counts[groups,None]).astype(int)
    idx=order[starts[groups,None]+pick]
    save_npz(folder/"schedule.npz",anchors=anchors.astype(np.int32),positive=idx.astype(np.int32))
    prior=Prior.load(HERE/"prior.npz")
    sc=np.load(HERE/"scaling.npz")
    cache=np.load(HERE/"positive.npy",mmap_mode="r")
    angular=arm!="radial"
    slices=blocks(angular=angular)
    dim=slices["power0"].stop if arm=="power0" else max(v.stop for v in slices.values())
    mu,sd=[torch.as_tensor(sc[key][:dim],device=device) for key in ("mean","scale")]
    model=new_model(arm).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    start=0;past_wall=past_cpu=0.;trace=[]
    for u in reversed(MILESTONES):
        ck=folder/f"update-{u:05}.pt"
        if u>stop or not ck.exists():continue
        info=json.loads(ck.with_suffix(".json").read_text())
        if digest(ck)!=info["checkpoint_sha256"]:raise ValueError("Checkpoint hash mismatch")
        if u==stop:return
        state=torch.load(ck,map_location=device,weights_only=True)
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"])
        start=u;past_wall=info["wall_seconds"];past_cpu=info["worker_cpu_seconds"];trace=info["trace"]
        break
    clock=time.perf_counter();cpu=time.process_time()
    for step in range(start,stop):
        indices=idx[step].flatten(); zz=z[indices]
        # Streams distinguished by fit seed, optimizer update and bank role, reused across arms.
        rngq=np.random.default_rng(np.random.SeedSequence([234901,seed,step,0]))
        rngn=np.random.default_rng(np.random.SeedSequence([234901,seed,step,1]))
        def tensor(a):return torch.as_tensor(a,device=device)
        zz_t=tensor(zz.astype(np.int64)); kq,fq=[tensor(x) for x in prior.draw(zz,rngq)]
        if arm=="flow16":
            tgen=torch.Generator(device=device).manual_seed(seed*100000+step)
            loss=flow_loss(model,zz_t,kq,fq,tensor(k[indices].astype(np.float32)),
                           tensor(f[indices].astype(np.float32)),tgen)
        else:
            with torch.no_grad():
                kn,fn=[tensor(x) for x in prior.draw(zz,rngn)]
                L,fp=geometry(model,arm,zz_t,kn,fn)
                neg=(features(L,fp,zz_t,angular=angular,**DESCRIPTOR)[:,:dim]-mu)/sd
                pos=(tensor(cache[indices,:dim])-mu)/sd
            L,fp=geometry(model,arm,zz_t,kq,fq)
            q=(features(L,fp,zz_t,angular=angular,**DESCRIPTOR)[:,:dim]-mu)/sd
            shape=(GROUPS,BANK,dim)
            d=within_direction(q.detach().reshape(shape),pos.reshape(shape),neg.reshape(shape))
            loss=.5*(q-q.detach()-d.reshape_as(q)).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        opt.step()
        u=step+1
        if u in MILESTONES or u%64==0:trace.append([u,float(loss.detach()),float(norm)])
        if u not in MILESTONES:continue
        path=folder/f"update-{u:05}.pt";tmp=path.with_suffix(".building")
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=arm,seed=seed,updates=u,
                        prior_sha256=digest(HERE/"prior.npz")),tmp);tmp.replace(path)
        bcounts=(np.diff(np.sort(idx[:u],axis=-1),axis=-1)!=0).sum(-1)+1
        meta=dict(arm=arm,seed=seed,updates=u,parameters=sum(p.numel() for p in model.parameters()),
                schedule_anchors=int(u*GROUPS),distinct_positive_records=int(len(np.unique(idx[:u]))),
                positive_examples=u*GROUPS*BANK,bank_distinct_histogram=np.bincount(bcounts.ravel(),minlength=5).tolist(),
                singleton_frac=float((counts[groups[:u]]==1).mean()),
                seconds_per_update=(time.perf_counter()-clock)/(u-start),wall_seconds=past_wall+time.perf_counter()-clock,
                worker_cpu_seconds=past_cpu+time.process_time()-cpu,
                device=device,torch=torch.__version__,host_threads=1,
                peak_working_set_bytes=psutil.Process().memory_info()._asdict().get("peak_wset",None),
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else 0,
                optimizer="AdamW lr=0.001 default betas and decay, grad norm cap 10",trace=trace,
                checkpoint_sha256=digest(path),schedule_sha256=digest(folder/"schedule.npz"),
                prior_sha256=digest(HERE/"prior.npz"),scaling_sha256=digest(HERE/"scaling.npz"),
                train_sha256=digest(trainpath),implementation={p.name:digest(p) for p in
                     (HERE/"model.py",HERE/"train.py",HERE.parent/"physical.py",HERE.parent/"neighbors.py",ROOT/"scaled/family.py")})
        write_json(path.with_suffix(".json"),meta)
        print(arm,seed,u,round(meta["seconds_per_update"],3),"s/update",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    p.add_argument("--stop",type=int,choices=MILESTONES,default=3392)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    jobs=[(arm,seed,a.stop,a.device) for seed in a.seeds for arm in a.arms]
    if a.workers==1:
        for job in jobs:train(job)
    else:
        with ProcessPoolExecutor(a.workers) as pool:list(pool.map(train,jobs))
