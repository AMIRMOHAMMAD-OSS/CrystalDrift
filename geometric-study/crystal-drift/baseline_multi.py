"""Permutation-equivariant periodic flow matching for binary cells of 2–8 sites.

The split, optimizer, and train-only lattice chart follow baseline.py. A
masked DeepSets velocity uses coordinate Fourier moments relative to each
site separately for the two chemical species. A common fractional shift
cancels exactly from those features. Subtracting the mean wrapped training
displacement recovers the target configuration up to global translation, as
in the translation-corrected torus path of Miller et al., FlowMM (ICML 2024).
"""
import argparse
import json
import math
import statistics
import time

import numpy as np
import torch
from torch import nn

from baseline import HERE, TORUS_SCALE, periodic_difference


class MultiVelocity(nn.Module):
    def __init__(self, width=56):
        super().__init__()
        self.element = nn.Embedding(119, 16, padding_idx=0)
        # Global input: both elements, two counts, six log-cell entries, four time features.
        self.site = nn.Sequential(nn.Linear(44+16+24,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.position = nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,3))
        self.lattice = nn.Sequential(nn.Linear(width+44,width),nn.SiLU(),nn.Linear(width,6))

    def prepare(self,z):
        mask = z.ne(0)
        lo = z.masked_fill(~mask,119).amin(1)
        hi = z.amax(1)
        group = torch.stack((z==lo[:,None],z==hi[:,None]),dim=1).float()
        counts = group.sum(dim=2)
        group = group / counts.unsqueeze(-1)
        composition = torch.cat((self.element(lo),self.element(hi),counts/8),dim=1)
        return mask.unsqueeze(-1).float(), group, self.element(z), composition

    def forward(self,conditioning,k,f,t):
        mask,group,element,composition = conditioning
        time_features = torch.cat((t,1-t,torch.sin(math.pi*t),torch.cos(math.pi*t)),dim=1)
        global_features = torch.cat((composition,k,time_features),dim=1)
        phase = torch.cat((f,f*2),dim=2) * (2*math.pi)
        co,si = phase.cos(),phase.sin()
        # Multiplication by exp(-i phase_i) puts each circular mean in site i's frame.
        mc = torch.bmm(group,co)[:,None]
        ms = torch.bmm(group,si)[:,None]
        rel_re = co[:,:,None]*mc + si[:,:,None]*ms
        rel_im = co[:,:,None]*ms - si[:,:,None]*mc
        features = torch.cat((element,rel_re.flatten(2),rel_im.flatten(2),
                              global_features[:,None].expand(-1,f.shape[1],-1)),dim=2)
        h = self.site(features)*mask
        pooled = h.sum(1) / mask.sum(1)
        raw = self.position(torch.cat((h,pooled[:,None].expand_as(h)),dim=2))
        v = (raw-(raw*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True))*mask
        return self.lattice(torch.cat((pooled,global_features),dim=1)),v


def train(d,seed,updates,batch_size):
    torch.manual_seed(seed)
    model = MultiVelocity()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    z = torch.from_numpy(d["z"].astype(np.int64))
    f = torch.from_numpy(d["f"].astype(np.float32))
    raw = d["k"].astype(np.float64)
    mean = torch.from_numpy(raw.mean(0).astype(np.float32))
    sd = torch.from_numpy(np.maximum(raw.std(0),1e-6).astype(np.float32))
    k = (torch.from_numpy(raw.astype(np.float32))-mean)/sd
    history = []
    tic = time.perf_counter()
    for _ in range(updates):
        ix = torch.randint(len(z),(batch_size,))
        condition = model.prepare(z[ix])
        mask = condition[0]
        k0 = torch.randn_like(k[ix])
        f0 = torch.rand_like(f[ix])*mask
        t = torch.rand(batch_size,1)
        dk = k[ix]-k0
        df = periodic_difference(f[ix],f0)*mask
        df = (df-df.sum(1,keepdim=True)/mask.sum(1,keepdim=True))*mask
        kt = k0 + t*dk
        ft = torch.remainder(f0 + t[:,None]*df,1.)*mask
        vk,vf = model(condition,kt,ft,t)
        cell_loss = (vk-dk).square().mean(1)
        site_loss = (vf-TORUS_SCALE*df).square().sum((1,2))/(3*mask.sum((1,2)))
        loss = (cell_loss+site_loss).mean()/2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(loss.detach().item())
    return model.eval(),mean,sd,dict(training_seconds=time.perf_counter()-tic,
             first_100_loss=np.mean(history[:100]).item(),last_100_loss=np.mean(history[-100:]).item())


@torch.inference_mode()
def integrate(model,z,k,f,mean,sd,nfe):
    cond = model.prepare(z)
    mask=cond[0]
    f=f*mask
    for i in range(nfe):
        t=torch.full((len(z),1),i/nfe,dtype=k.dtype)
        vk,vf=model(cond,k,f,t)
        k=k+vk/nfe
        f=torch.remainder(f+vf/(nfe*TORUS_SCALE),1.)*mask
    return k*sd+mean,f


def run(args):
    torch.set_num_threads(1)
    out=HERE/"baseline-multi-results"
    out.mkdir(exist_ok=True)
    with np.load(HERE/"data"/"multi-train.npz") as f:
        train_data={k:f[k] for k in ("z","k","f","ids")}
    with np.load(HERE/"data"/"multi-test.npz") as f:
        test_z,ids=f["z"],f["ids"]
    assert set(ids).isdisjoint(set(train_data["ids"]))
    n=len(ids)
    draws=20
    z=torch.from_numpy(np.repeat(test_z.astype(np.int64),draws,axis=0))
    report=dict(train_count=len(train_data["ids"]),test_count=n,draws=draws,updates=args.updates,batch_size=64,
                threads=1,optimizer="AdamW",learning_rate=.001,weight_decay=.01,numpy=np.__version__,
                torch=torch.__version__,seeds={},timing="Median CPU sampling time over all held-out draws in chunks of 256, including composition embedding, Euler integration and raw lattice decoding. Excludes RNG, disk I/O and quality metrics.")
    for seed in args.seeds:
        model,mean,sd,stats=train(train_data,seed,args.updates,64)
        stats["parameters"]=sum(p.numel() for p in model.parameters())
        stats["sampling"]={}
        rng=np.random.default_rng(100000+seed)
        k0=torch.from_numpy(rng.normal(size=(n,draws,6)).astype(np.float32).reshape(-1,6))
        f0=torch.from_numpy(rng.random(size=(n,draws,8,3)).astype(np.float32).reshape(-1,8,3))
        for nfe in (1,16):
            integrate(model,z[:256],k0[:256],f0[:256],mean,sd,nfe)
            durations=[]
            for _ in range(args.timing_repeats):
                tic=time.perf_counter()
                all_k,all_f=[],[]
                for start in range(0,len(z),256):
                    sl=slice(start,start+256)
                    k,f=integrate(model,z[sl],k0[sl],f0[sl],mean,sd,nfe)
                    all_k.append(k)
                    all_f.append(f)
                k=torch.cat(all_k).reshape(n,draws,6).numpy()
                f=torch.cat(all_f).reshape(n,draws,8,3).numpy()
                durations.append(time.perf_counter()-tic)
            assert np.isfinite(k).all() and np.isfinite(f).all() and ((f>=0)&(f<1)).all()
            assert not np.any(f.transpose(0,2,1,3)[test_z==0])
            np.savez_compressed(out/f"seed{seed}-euler{nfe}.npz",k=k,f=f,z=test_z,ids=ids)
            one_times=[]
            for index in np.linspace(0,len(z)-1,32,dtype=int):
                sl=slice(index,index+1)
                if not one_times:
                    integrate(model,z[sl],k0[sl],f0[sl],mean,sd,nfe)
                tic=time.perf_counter()
                integrate(model,z[sl],k0[sl],f0[sl],mean,sd,nfe)
                one_times.append(time.perf_counter()-tic)
            median=statistics.median(durations)
            stats["sampling"][nfe]=dict(median_batch_seconds=median,crystals=n*draws,
                                 microseconds_per_crystal=1e6*median/(n*draws),repeats=len(durations),
                                 median_single_crystal_microseconds=1e6*statistics.median(one_times),
                                 single_crystal_repeats=len(one_times))
        torch.save(dict(model=model.state_dict(),mean=mean,scale=sd,seed=seed),out/f"seed{seed}.pt")
        report["seeds"][str(seed)]=stats
        (out/"run.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
        print(f"seed {seed} finished in {stats['training_seconds']:.1f} training seconds",flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    arg.add_argument("--updates",type=int,default=1200)
    arg.add_argument("--timing-repeats",type=int,default=3)
    run(arg.parse_args())
