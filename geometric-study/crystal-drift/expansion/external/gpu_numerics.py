"""Measure repeated-source variation and endpoint differences in five GPU updates."""
import argparse
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from check_sampler import author_adapter
from crystalflow import HERE,EncoderCache,load_model,sample
from joint_timing import propose
from timing import decode


def compare(a,b):
    z,x,L=a;zz,xx,LL=b
    return dict(atom_types_equal=bool(np.array_equal(z,zz)),cell_max_A=float(np.max(np.abs(L-LL))),
                fractional_max=float(np.max(np.abs(x-xx-np.rint(x-xx)))))


@torch.inference_mode()
def run(opts):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    r=[]
    for label,weights in (("conditional",opts.csp),("joint",opts.joint)):
        net,emb,build=load_model(opts.source,weights,"cuda")
        model,emb2,build2=load_model(opts.source,weights,"cuda")
        cache=EncoderCache(model)
        joint=net.pred_type
        native=author_adapter(opts.source,net,emb,build,joint)
        z=np.zeros((2,20),np.int16);z[0,:2]=[3,8];z[1,:4]=[8,8,14,14]
        n=np.array([2,4])
        batch=SimpleNamespace(num_graphs=2,num_nodes=6,num_atoms=torch.as_tensor(n,device="cuda"),
            atom_types=torch.as_tensor(z[z>0],device="cuda",dtype=torch.long),
            batch=torch.as_tensor(np.repeat(np.arange(2),n),device="cuda"))
        def orig():
            torch.manual_seed(900223)
            out,_=native.sample(batch,N=5)
            species=out["atom_types"].cpu().numpy();x=out["frac_coords"].cpu().numpy()
            zz=np.zeros_like(z);xx=np.zeros((2,20,3))
            for i,(a,b) in enumerate(((0,2),(2,6))):
                order=np.argsort(species[a:b],kind="stable")
                zz[i,:b-a]=species[a:b][order];xx[i,:b-a]=x[a:b][order]
            return zz,xx,out["lattices"].double().cpu().numpy()
        def fast():
            torch.manual_seed(900223)
            if joint:
                zz,k,x,L,_,_=propose((model,emb2,build2,cache),n,5)
            else:
                k,x=sample(model,emb2,build2,cache,z,5,"cuda",None)
                zz,L,_,_=decode(z,k,x)
            return zz,x,L
        ieee={}
        for tf32 in (False,True):
            torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
            for deterministic in (False,True):
                torch.use_deterministic_algorithms(deterministic)
                original1,original2,fast1,fast2=orig(),orig(),fast(),fast()
                entry=dict(model=label,tf32=tf32,deterministic=deterministic,
                    source_repeat=compare(original1,original2),portable_repeat=compare(fast1,fast2),
                    source_vs_portable=compare(original1,fast1))
                if tf32:
                    o,p=ieee[deterministic]
                    entry["source_tf32_vs_ieee"]=compare(o,original1)
                    entry["portable_tf32_vs_ieee"]=compare(p,fast1)
                else:ieee[deterministic]=(original1,fast1)
                r.append(entry)
    (HERE/"gpu-source-numerics.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("--source","--csp","--joint"):p.add_argument(arg,type=Path,required=True)
    run(p.parse_args())
