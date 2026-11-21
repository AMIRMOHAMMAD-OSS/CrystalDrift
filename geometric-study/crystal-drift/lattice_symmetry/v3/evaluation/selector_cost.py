"""Time one complete formula shortlist and its auxiliary ranking on a warmed, single CPU thread."""
import json
import os
from pathlib import Path
from time import perf_counter_ns,perf_counter
import sys
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from lattice_symmetry.v3.evaluation.predict import setup_split,fit_selector,atomic_npz
from lattice_symmetry.v3.models.sampler import Sampler

@torch.inference_mode()
def main():
    import resource
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tic=perf_counter();s=Sampler();index=perf_counter()-tic
    rss1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ranks={k:fit_selector(s,k) for k in ("frequency","learned")};names=["lookup","frequency","learned"]
    calls=[0]
    def hook(module,args,out):calls[0]+=1
    ranks["learned"].model.register_forward_hook(hook)
    data,_=setup_split("clock");order=np.random.default_rng(2428)
    def propose(kind,z):
        s.shortlist.cache_clear()
        t=perf_counter_ns()
        active=np.asarray(z)[z>0]
        options=s.shortlist(tuple(np.sort(active)),False) if kind=="lookup" else ranks[kind].rank(active,kind)
        return (perf_counter_ns()-t)*1e-6,len(options)
    for name in names:
        for _,_,z in data[:3]:propose(name,z)
    calls[0]=0
    times=np.zeros((len(data),3));lengths=np.zeros_like(times,dtype=np.int16);seq=[]
    for i,(_,_,z) in enumerate(data):
        visit=order.permutation(3);seq.append(visit)
        for j in visit: times[i,j],lengths[i,j]=propose(names[j],z)
    assert calls[0]==len(data)
    assert np.array_equal(lengths[:,0],lengths[:,1]) and np.array_equal(lengths[:,0],lengths[:,2])
    dst=HERE/"timing";dst.mkdir(exist_ok=True)
    atomic_npz(dst/"selector-cost.npz",ids=[m for m,_,z in data],names=names,ms=times,shortlist_lengths=lengths,
        order=seq,index_seconds=index,index_peak_delta_kib=rss1-rss0,calls=calls[0],threads=1)
    print({name:np.percentile(times[:,i],[50,95]).tolist() for i,name in enumerate(names)},flush=True)
if __name__=="__main__":main()
