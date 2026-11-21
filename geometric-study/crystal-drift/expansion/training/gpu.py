"""Matched FP32 GPU proposal timings on 128 multielement MP-20 compositions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"

import numpy as np
import torch

from controls import ROOT,build,generate
from evaluate import Retrieval
from experiment import load
from latency import decode

HERE=Path(__file__).resolve().parent
GPU=HERE/"gpu"
EXT=ROOT/"revision/full/external"


def official():
    spec=importlib.util.spec_from_file_location("pinned_uflow",EXT/"generate.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def published_model(weights,temp):
    mod=official();mod.official_sources(temp)
    from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import MeanFlowSiTRawFormulaAtomwiseSystem,build_sg_to_crystal_system
    from src.models.meanflow_transport import MeanFlowTransport
    with torch.device("meta"):
        model=MeanFlowSiTRawFormulaAtomwiseSystem(input_size=100,atom_type_vocab_size=120,hidden_size=768,
            depth=12,num_heads=12,predict_atom_types=False,use_spacegroup_conditioning=False,
            use_formula_embedding=True,use_atomwise_features=True,use_crystal_system_embedding=True)
    model.load_state_dict(mod.safe_weights(weights),strict=True,assign=True)
    with np.load(EXT/"chemistry.npz") as chemistry:
        model.atomic_feature_table=torch.from_numpy(chemistry["table"].copy())
        key=chemistry["sort_key"]
    model.sg_to_crystal_system=build_sg_to_crystal_system(230)
    model=model.to("cuda").eval()
    transport=MeanFlowTransport(torus_coords=False,lattice_per_structure=False,proj_lattice=False,
                                proj_coords_anchor=False,proj_coords_rotavg=False)
    return mod,model,transport,key


def service(method,compositions,weights,temp):
    if method=="retrieval":
        train=load(ROOT/"scaled/mp20-data/train.npz")
        bank=Retrieval(train)
        donors={tuple(z):bank.pool(z) for z in compositions}
        def propose(z,rng):
            ix=[rng.choice(donors[tuple(row)]) for row in z]
            return train["k"][ix],train["f"][ix]
    elif method=="uflow":
        mod,model,transport,key=published_model(weights,temp)
        tables={}
        for row in compositions:
            token=np.zeros(100,dtype=np.int64)
            atoms=row[row>0]
            token[:len(atoms)]=sorted(atoms,key=lambda a:(key[a],a))
            tables[tuple(row)]=token
        def propose(z,rng):
            batch=len(z)
            a=torch.as_tensor(np.stack([tables[tuple(row)] for row in z]),device="cuda")
            prior=transport.sample_prior(batch,100,12,"cpu").to("cuda")
            cond=dict(atom_types=a,mask=a!=0,spacegroup=torch.zeros(batch,dtype=torch.long,device="cuda"),
                      generate_atom_types=False)
            return mod.structures(transport.sample(model,prior,cond,num_steps=1),a,z)
    else:
        name="flow" if method.startswith("flow") else method
        selected=json.loads((HERE/"selection.json").read_text())[name]["pass"]
        path=HERE/f"fits/{name}-seed11/pass{selected}.pt"
        state=torch.load(path,weights_only=True,map_location="cpu")
        model=build(name).to("cuda").eval()
        model.load_state_dict(state["model"])
        mean,scale=state["mean"].to("cuda"),state["scale"].to("cuda")
        steps=16 if method=="flow16" else 1
        def propose(z,rng):
            a=torch.as_tensor(z,device="cuda",dtype=torch.long)
            k=torch.randn((len(z),6),device="cuda")
            f=torch.rand((*z.shape,3),device="cuda")
            y,x=generate(model,name,a,k,f,steps)
            return (y*scale+mean).cpu().numpy(),x.cpu().numpy()
    def complete(z,rng):
        k,f=propose(z,rng)
        f=np.where(z[:,:,None]>0,np.remainder(f,1),0.).astype(np.float64)
        cell,f,cart=decode(k,f)
        return z.copy(),cell,f,cart,k
    return complete


@torch.inference_mode()
def measure(method,ids,z,args,temp):
    if (GPU/f"{method}-10000.npz").exists() and (GPU/f"{method}-episodes.npz").exists():return
    fun=service(method,z,args.weights,temp)
    for mode in ("episodes","10000"):
        path=GPU/f"{method}-{mode}.npz"
        if path.exists():continue
        batch=1 if mode=="episodes" else args.batch
        rng=np.random.default_rng(13651)
        torch.manual_seed(13651)
        for _ in range(5):fun(z[:batch],rng)
        seed=554011 if mode=="episodes" else 661011
        rng=np.random.default_rng(seed);torch.manual_seed(seed);torch.cuda.synchronize()
        if mode=="episodes":
            k=np.empty((len(ids),20,6));f=np.empty((len(ids),20,20,3));elapsed=np.empty((len(ids),20))
            for i in range(len(ids)):
                out=[];tic=time.perf_counter_ns()
                for draw in range(20):
                    proposal=fun(z[i:i+1],rng)
                    done=time.perf_counter_ns()
                    out.append((proposal,done))
                for draw,(p,t) in enumerate(out):k[i,draw],f[i,draw],elapsed[i,draw]=p[4][0],p[2][0],(t-tic)*1e-6
            data=dict(ids=ids,z=z,k=k,f=f,elapsed_ms=elapsed)
        else:
            index=np.arange(10000)%len(ids);duration=[];counts=[]
            tic=time.perf_counter_ns()
            for i in range(0,len(index),batch):
                start=time.perf_counter_ns()
                p=fun(z[index[i:i+batch]],rng)
                duration.append((time.perf_counter_ns()-start)*1e-6)
                counts.append(len(p[0]))
            seconds=(time.perf_counter_ns()-tic)*1e-9
            assert sum(counts)==10000 and np.isfinite(p[3]).all()
            data=dict(ids=ids[index],elapsed_s=seconds,batch_ms=duration,counts=counts)
        tmp=path.with_suffix(".building")
        with tmp.open("wb") as fp:np.savez_compressed(fp,**data)
        tmp.replace(path)
        print(method,mode,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights",type=Path);ap.add_argument("--site",nargs="*",type=Path,default=[])
    ap.add_argument("--methods",nargs="+",default=["drift","iid","meanflow","flow1","flow16","retrieval","uflow"])
    ap.add_argument("--batch",type=int,default=24)
    args=ap.parse_args()
    sys.path.extend(map(str,args.site))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    GPU.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as cohort:ids,z=cohort["ids"][:128],cohort["z"][:128]
    config=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),
        batch_size=args.batch,single_batch_size=1,host_threads=1,tf32=False,model_dtype="float32",output_dtype="float64",
        output="CPU species, canonical lattice, wrapped fractional and Cartesian coordinates",warmups=5,workload=10000,
        fitted_seed=11,max_proposals_per_episode=20,episodes=len(ids),selection="expansion/training/selection.json",
        excluded=["model loading","donor index construction","retrospective scoring","relaxation"])
    (GPU/"config.json").write_text(json.dumps(config,indent=1)+"\n")
    with tempfile.TemporaryDirectory() as td:
        for m in args.methods:
            measure(m,ids,z,args,Path(td))
            torch.cuda.empty_cache()


if __name__=="__main__":main()
