"""Conditional periodic flow matching for two-site AB crystals.

The fixed composition is ordered by atomic number. The two atoms occupy 0 and
f on the fractional torus; k holds the six components of half the matrix log
of the cell Gram matrix. Source lattice coordinates are standard normal in
training-set standardized units. The fractional source is Haar uniform.

This compact reference fits conditional geodesic flow matching on the product
space. The periodic path follows FlowMM (Miller et al., ICML 2024,
https://proceedings.mlr.press/v235/miller24a.html). Architecture, lattice
chart, restricted task, and fitted parameters are local to this pilot.
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
TORUS_SCALE = math.sqrt(12)


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.element = nn.Embedding(119, 16)
        self.network = nn.Sequential(nn.Linear(48, 96), nn.SiLU(),
                                     nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 9))

    def forward(self, z, lattice, fraction, t):
        phase = 2 * math.pi * fraction
        time_code = torch.cat((t, 1-t, torch.sin(math.pi*t), torch.cos(math.pi*t)),dim=1)
        features = torch.cat((self.element(z).flatten(1), lattice, phase.cos(),
                              phase.sin(), time_code), dim=1)
        return self.network(features)


def periodic_difference(target, origin):
    return torch.remainder(target - origin + 0.5, 1.0) - 0.5


def train(train_data, seed, updates, batch_size):
    torch.manual_seed(seed)
    model = Velocity()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    z = torch.from_numpy(train_data["z"].astype(np.int64))
    f = torch.from_numpy(train_data["f"].astype(np.float32))
    raw_k = np.asarray(train_data["k"],dtype=np.float64)
    mean = torch.from_numpy(raw_k.mean(0).astype(np.float32))
    scale = torch.from_numpy(np.maximum(raw_k.std(0),1e-6).astype(np.float32))
    k = (torch.from_numpy(raw_k.astype(np.float32)) - mean) / scale

    losses = []
    tic = time.perf_counter()
    for step in range(updates):
        indices = torch.randint(len(z), (batch_size,))
        target_k, target_f = k[indices], f[indices]
        start_k = torch.randn_like(target_k)
        start_f = torch.rand_like(target_f)
        t = torch.rand(batch_size, 1)
        dk = target_k - start_k
        df = periodic_difference(target_f, start_f)
        kt = start_k + t * dk
        ft = torch.remainder(start_f + t * df, 1.)
        velocity = model(z[indices],kt,ft,t)
        # sqrt(12) normalizes uniform circular displacement to unit variance.
        target_velocity = torch.cat((dk, TORUS_SCALE*df),dim=1)
        loss = nn.functional.mse_loss(velocity,target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter()-tic
    return model.eval(),mean,scale,dict(training_seconds=elapsed,
              first_100_loss=np.mean(losses[:100]).item(),last_100_loss=np.mean(losses[-100:]).item())


def eval_source(z, draws, seed):
    rng = np.random.default_rng(100000+seed)
    k = rng.normal(size=(len(z),draws,6)).astype(np.float32)
    f = rng.random(size=(len(z),draws,3)).astype(np.float32)
    return (torch.from_numpy(np.repeat(z.astype(np.int64),draws,axis=0)),
            torch.from_numpy(k.reshape(-1,6)),torch.from_numpy(f.reshape(-1,3)))


@torch.inference_mode()
def integrate(model,source,mean,scale,steps):
    z,k,f = source
    for j in range(steps):
        t = torch.full((len(z),1),j/steps,dtype=k.dtype)
        v = model(z,k,f,t)
        k = k + v[:,:6]/steps
        f = torch.remainder(f + v[:,6:]/(steps*TORUS_SCALE),1.)
    return k*scale+mean,f


def run(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(exist_ok=True,parents=True)
    with np.load(args.data/"train.npz") as f:
        train_data = {k:f[k] for k in ("z","f","k","ids")}
    with np.load(args.data/"test.npz") as f:
        test_z,test_ids = f["z"],f["ids"]
    assert set(train_data["ids"]).isdisjoint(set(test_ids))
    assert np.all(train_data["z"][:,0] < train_data["z"][:,1])
    assert np.all(test_z[:,0]<test_z[:,1])
    metrics = dict(torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
                   threads=args.threads,updates=args.updates,batch_size=args.batch_size,draws=args.draws,
                   train_count=len(train_data["z"]),test_count=len(test_z),optimizer="AdamW",
                   learning_rate=0.001,weight_decay=0.01,seeds={},timing="Warm CPU forward and Euler integration, including output unstandardization. Reports batched throughput and single-crystal latency. Excludes source RNG, file I/O, and quality metrics.")
    for seed in args.seeds:
        model,mean,scale,training = train(train_data,seed,args.updates,args.batch_size)
        source = eval_source(test_z,args.draws,seed)
        data = dict(**training,parameters=sum(p.numel() for p in model.parameters()),sampling={})
        for nfe in args.steps:
            integrate(model,source,mean,scale,nfe)
            durations = []
            for _ in range(args.timing_repeats):
                tic = time.perf_counter()
                k,f = integrate(model,source,mean,scale,nfe)
                durations.append(time.perf_counter()-tic)
            elapsed=statistics.median(durations)
            k = k.reshape(len(test_z),args.draws,6).numpy()
            f = f.reshape(len(test_z),args.draws,3).numpy()
            assert np.isfinite(k).all() and np.isfinite(f).all() and ((0<=f)&(f<1)).all()
            np.savez_compressed(args.output/f"seed{seed}-euler{nfe}.npz",k=k,f=f,z=test_z,ids=test_ids)
            single_times = []
            for index in np.linspace(0,len(source[0])-1,min(32,len(source[0])),dtype=int):
                one=tuple(x[index:index+1] for x in source)
                if not single_times:
                    integrate(model,one,mean,scale,nfe)
                tic=time.perf_counter()
                integrate(model,one,mean,scale,nfe)
                single_times.append(time.perf_counter()-tic)
            data["sampling"][nfe]=dict(batch_crystals=k.size//6,median_batch_seconds=elapsed,
                      microseconds_per_crystal=elapsed*1e6/(len(test_z)*args.draws),repeats=len(durations),
                      median_single_crystal_microseconds=statistics.median(single_times)*1e6,
                      single_crystal_repeats=len(single_times))
        torch.save(dict(model=model.state_dict(),mean=mean,scale=scale,seed=seed),args.output/f"seed{seed}.pt")
        metrics["seeds"][str(seed)] = data
        (args.output/"run.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf8")
        print(f"seed {seed}: {data['parameters']} parameters, {data['training_seconds']:.1f} train seconds",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=HERE/"data")
    p.add_argument("--output",type=Path,default=HERE/"baseline-results")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    p.add_argument("--steps",type=int,nargs="+",default=[1,4,16])
    p.add_argument("--draws",type=int,default=20)
    p.add_argument("--updates",type=int,default=1200)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--timing-repeats",type=int,default=7)
    p.add_argument("--threads",type=int,default=1)
    run(p.parse_args())
