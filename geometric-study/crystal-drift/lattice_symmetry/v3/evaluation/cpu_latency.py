"""CPU batch-one first-proposal timing with real lookup, one host thread and warm networks."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
from time import perf_counter,perf_counter_ns
for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[var]="1"
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.evaluation.predict import setup_split,atomic_npz,h
from lattice_symmetry.v3.evaluation.paired import checkpoint_for,seed_at

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("matrix",type=Path);ap.add_argument("--uflow",action="store_true")
    a=ap.parse_args();specs=json.loads(a.matrix.read_text())["methods"]
    if not a.uflow:specs=[v for v in specs if "u_tier" not in v]
    from lattice_symmetry.v3.models.sampler import Sampler
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision("highest")
    t=perf_counter();s=Sampler();index_s=perf_counter()-t
    devices={};counts={}
    if a.uflow:
        from lattice_symmetry.v3.uflow import runtime
        flow=runtime.transport()
    for v in specs:
        tag=v["tag"];net=None
        if "u_tier" in v:
            net=runtime.model(v["u_tier"],"cpu").eval()
            if v["u_tier"]!="released":net.load_state_dict(torch.load(checkpoint_for(v),map_location="cpu",weights_only=True))
        elif v.get("checkpoint"):net=s.load(checkpoint_for(v))
        devices[tag]=net;counts[tag]=0
        if net is not None:
            def add(net,args,value,k=tag):counts[k]+=1
            net.register_forward_hook(add)
    cohort,_=setup_split("clock")
    def proposed(v,z,seed):
        s.shortlist.cache_clear();torch.manual_seed(seed)
        t=perf_counter_ns();zz=np.array(z,copy=True)
        if "u_tier" in v:L,F,Z=runtime.propose(devices[v["tag"]],flow,zz[None])
        else:
            active=zz[zz>0];opts=s.shortlist(tuple(np.sort(active)),False)
            sd=seed_at(s,active,opts[0],np.random.default_rng(seed),v.get("family")=="donor")
            L,F,Z=s.apply(devices[v["tag"]],[sd])[0]
        return (perf_counter_ns()-t)*1.e-6
    for v in specs:
        for i in range(4):proposed(v,cohort[i][2],93+i)
    for k in counts:counts[k]=0
    order=[];ms=np.zeros((len(cohort),len(specs)))
    rng=np.random.default_rng(61121)
    for i,(_,_,z) in enumerate(cohort):
        seq=rng.permutation(len(specs));order.append(seq)
        for j in seq:ms[i,j]=proposed(specs[j],z,68871+i)
    out=HERE/"timing";out.mkdir(exist_ok=True)
    n="cpu-first" if a.uflow else "cpu-compact"
    atomic_npz(out/(n+".npz"),ids=[r[0] for r in cohort],methods=[v["tag"] for v in specs],
        first_ms=ms,order=order,threads=1,torch=str(torch.__version__),index_seconds=index_s,calls=json.dumps(counts),
        platform=platform.platform(),matrix_sha=h(a.matrix))
    print({v["tag"]:np.percentile(ms[:,i],[50,95]).tolist() for i,v in enumerate(specs)},flush=True)
if __name__=="__main__":main()
