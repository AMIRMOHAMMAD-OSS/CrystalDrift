"""Measure the native small-width overhead and algebraic rank limit of the velocity head."""
import json
import time
import numpy as np
import torch
from runtime import HERE, TIERS, atomic_json, counts, model, propose, transport

def main():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    with np.load(HERE/"train-cache/00000.npz") as d:
        row=d["z"][0];sorted_z=np.pad(np.sort(row[row>0]),(0,sum(row==0)))
    flow=transport()
    result=[]
    for d,depth,heads in ((2,9,1),(4,1,1),(12,1,1),(64,2,4),(160,4,8)):
        TIERS["probe"]=(d,depth,heads)
        net=model("probe","cpu").eval()
        measured=counts(net)
        formula=3852+940*d+5*d*d+depth*(18*d*d+15*d)
        assert measured["registered"]==formula
        assert net.final_layer.linear.in_features==d and net.final_layer.linear.out_features==12
        for _ in range(3):
            propose(net,flow,sorted_z)
        elapsed=[]
        for _ in range(15):
            start=time.perf_counter_ns()
            proposal=propose(net,flow,sorted_z)
            elapsed.append((time.perf_counter_ns()-start)/1e6)
        assert np.isfinite(proposal[0]).all() and np.isfinite(proposal[1]).all()
        result.append(dict(width=d,depth=depth,heads=heads,**measured,max_velocity_head_rank=min(d,12),
                           median_cpu_ms=float(np.median(elapsed)),p95_cpu_ms=float(np.percentile(elapsed,95))))
    atomic_json(HERE/"capacity.json",dict(batch=1,threads=1,trials=15,warmup=3,
        measured="initial weights on a training formula, includes host lattice and fractional decoding",
        inactive_component="231-row fine space group embedding",model_size_formula="3852+940*d+5*d^2+depth*(18*d^2+15*d)",tiers=result))
if __name__=="__main__":main()
