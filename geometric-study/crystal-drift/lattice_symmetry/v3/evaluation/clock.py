"""Interleave unranked, synchronous, batch-one proposal streams on one GPU, then score offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter,perf_counter_ns
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.evaluation.predict import setup_split,atomic_npz,h,array
from lattice_symmetry.v3.evaluation.paired import checkpoint_for,seed_at
from lattice_symmetry.v3.models.sampler import Sampler
from lattice_symmetry.v3.uflow import runtime

def restored(spec):
    names={name:runtime.model(name,"cuda").eval() for name in set(v["u_tier"] for v in spec if "u_tier" in v)}
    for v in spec:
        if v.get("u_tier") not in (None,"released"):
            obj=torch.load(checkpoint_for(v),map_location="cpu",weights_only=True)
            weights=obj.get("ema",obj.get("net",obj))
            names[v["u_tier"]].load_state_dict(weights)
    return names

@torch.inference_mode()
def run(matrix,throughput=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    specs=json.loads(matrix.read_text())["methods"]
    assert torch.cuda.is_available() and json.loads(matrix.read_text()).get("frozen")
    out=HERE/"timing";out.mkdir(exist_ok=True)
    begin=perf_counter();s=Sampler("cuda");build=perf_counter()-begin
    unets=restored(specs);transport=runtime.transport()
    nets={v["tag"]:s.load(checkpoint_for(v)) for v in specs if v.get("checkpoint") and not v.get("u_tier")}
    counts={}; handles=[]
    def hook(key):
        def applied(*unused):counts[key]+=1
        return applied
    for v in specs:
        tag=v["tag"];counts[tag]=0
        net=unets[v["u_tier"]] if "u_tier" in v else nets.get(tag)
        if net is not None:handles.append(net.register_forward_hook(hook(tag)))
    cohort,_=setup_split("clock");order=np.random.default_rng(61877)
    result={v["tag"]:array(len(cohort),20) for v in specs}
    times={v["tag"]:np.zeros((len(cohort),20)) for v in specs};starts={k:[] for k in times}
    sequence=[]
    def episode(v,z,seed,k):
        tag=v["tag"];rng=np.random.default_rng(seed);torch.manual_seed(seed)
        s.shortlist.cache_clear()
        torch.cuda.synchronize()
        tick=perf_counter_ns();allout=[];completion=[]
        zz=np.array(z,copy=True);active=zz[zz>0]
        if "u_tier" not in v:choices=s.shortlist(tuple(np.sort(active)),False)
        for j in range(k):
            if "u_tier" in v:
                L,F,Z=runtime.propose(unets[v["u_tier"]],transport,zz[None])
                mask=Z[0]>0;geom=(L[0],F[0,mask],Z[0,mask])
            else:
                sd=seed_at(s,active,choices[j%len(choices)],rng,v.get("family")=="donor")
                geom=s.apply(nets.get(tag),[sd])[0]
            torch.cuda.synchronize();completion.append((perf_counter_ns()-tick)*1e-6)
            allout.append(geom)
        return allout,completion
    for v in specs:
        for j in range(3):episode(v,cohort[j][2],262+j,2)
    for k in counts:counts[k]=0
    torch.cuda.reset_peak_memory_stats()
    if throughput:
        with np.load(ROOT/"scaled/mp20-data/test.npz") as data:
            all_ids=data["ids"];all_z=data["z"]
        seq=np.resize(np.random.default_rng(51234).permutation(len(all_ids)),10000)
        for v in specs:
            tag=v["tag"];rng=np.random.default_rng(81836);torch.manual_seed(81836)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            allocated=torch.cuda.memory_allocated();tic=perf_counter();batch_times=[];total_atoms=0
            for at in range(0,10000,32):
                stamp=perf_counter();zz=all_z[seq[at:at+32]].copy()
                if "u_tier" in v:
                    L,F,Z=runtime.propose(unets[v["u_tier"]],transport,zz)
                    total_atoms+=int((Z>0).sum())
                else:
                    seeds=[]
                    for z in zz:
                        s.shortlist.cache_clear();active=z[z>0]
                        opts=s.shortlist(tuple(np.sort(active)),False)
                        seeds.append(seed_at(s,active,opts[0],rng,v.get("family")=="donor"))
                    outs=s.apply(nets.get(tag),seeds)
                    total_atoms+=sum(len(e) for l,f,e in outs)
                torch.cuda.synchronize();batch_times.append(perf_counter()-stamp)
            elapsed=perf_counter()-tic
            assert total_atoms==int((all_z[seq]>0).sum())
            assert counts[tag]==(len(batch_times) if tag in nets or "u_tier" in v else 0)
            atomic_npz(out/(tag+"-10k.npz"),ids=all_ids[seq],batch_seconds=batch_times,
                       total_seconds=elapsed,batch=32,atoms=total_atoms,calls=counts[tag],
                       resident_bytes=allocated,peak_additional_bytes=torch.cuda.max_memory_allocated()-allocated,
                       matrix_sha=h(matrix),device=torch.cuda.get_device_name(),index_seconds=build)
            print(tag,"10,000 end-to-end proposals",elapsed,flush=True)
        return
    import time
    for ix,(mid,_,z) in enumerate(cohort):
        visit=order.permutation(len(specs));sequence.append(visit.tolist())
        for j in visit:
            v=specs[j];tag=v["tag"];starts[tag].append(time.time())
            batch,elapsed=episode(v,z,98377+ix,20)
            times[tag][ix]=elapsed
            L,F,Z=result[tag]
            for b,(l,f,e) in enumerate(batch):
                L[ix,b]=l;F[ix,b,:len(e)]=f;Z[ix,b,:len(e)]=e
        if (ix+1)%16==0:print(ix+1,"clock targets",flush=True)
    for v in specs:
        tag=v["tag"];assert counts[tag]==len(cohort)*20*(1 if ("u_tier" in v or tag in nets) else 0),(tag,counts[tag])
        L,F,Z=result[tag]
        atomic_npz(out/(tag+".npz"),ids=[r[0] for r in cohort],L=L,F=F,Z=Z,elapsed_ms=times[tag],starts=starts[tag],calls=counts[tag])
    info=dict(matrix_sha=h(matrix),device=torch.cuda.get_device_name(),pytorch=str(torch.__version__),
        platform=platform.platform(),cpu_threads=1,precision="FP32, TF32 disabled",index_seconds=build,peak_all_models_bytes=torch.cuda.max_memory_allocated(),
        batch_size=1,cap=20,order=sequence,method_order=[s["tag"] for s in specs],calls=counts,
        boundary="after formula processing, exact shortlist lookup anew per episode, seed construction, first GPU transfer, forward, native-cell/fractional/element host array materialization and CUDA sync; initialization, disk & matcher excluded")
    (out/"settings.json").write_text(json.dumps(info,indent=2))
    for handle in handles:handle.remove()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("matrix",type=Path);parser.add_argument("--throughput",action="store_true")
    options=parser.parse_args();run(options.matrix,options.throughput)

