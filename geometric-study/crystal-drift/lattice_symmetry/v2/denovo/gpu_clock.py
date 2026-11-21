"""Time chemistry-to-host generation on the CUDA device and replay the archived noise stream."""
import argparse
import os
from pathlib import Path
import sys
from time import perf_counter
for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"

import numpy as np
import torch
from generate_pools import HERE, ROOT, CompositionPrior, cells, draw, fitted, setup


def geometry(arm):
    if arm!="fixed-basis":
        return fitted.load_service(arm,11,"cuda"),fitted.checkpoint_path(arm,11)
    model,state,_,path=setup("drift","cuda")
    @torch.inference_mode()
    def propose(z,rng):
        k=rng.standard_normal((len(z),6)).astype(np.float32)
        f=rng.random((len(z),20,3),dtype=np.float32)
        at=torch.as_tensor(z,device="cuda",dtype=torch.long)
        k,f=draw(model,"drift",at,torch.from_numpy(k).cuda(),torch.from_numpy(f).cuda(),1)
        return (k*state["scale"]+state["mean"]).cpu().numpy(),f.cpu().numpy()
    return propose,path


def run(arm):
    with np.load(ROOT/"scaled/mp20-data/train.npz") as data:
        prior=CompositionPrior(data["z"],shrinkage=4.)
    propose,path=geometry(arm)
    pool=HERE/"pools"/(arm+".npz")
    with np.load(pool) as data:z0,k0,f0=(data[key] for key in ("z","k","f"))
    def sample(n,chem,noise):
        z=prior.sample(chem,n,"learned")
        k,f=propose(z,noise)
        lattice=cells(k)
        cart=f@lattice
        finite=np.isfinite(k).all(1)&np.isfinite(f).all((1,2))&np.isfinite(cart).all((1,2))
        return z,k,f,finite
    warm=np.random.default_rng(112926)
    for _ in range(3):sample(16,warm,warm)
    chem=np.random.default_rng(22112026)
    noise=np.random.default_rng(71831)
    times=[];sizes=[];dk=0.;df=0.;nfinite=0
    for i in range(0,10000,128):
        n=min(128,10000-i);tic=perf_counter()
        z,k,f,valid=sample(n,chem,noise)
        times.append(perf_counter()-tic);sizes.append(n)
        np.testing.assert_equal(z,z0[i:i+n])
        nfinite+=int(valid.sum())
        dk=max(dk,float(np.abs(k-k0[i:i+n]).max()))
        diff=f-f0[i:i+n]
        df=max(df,float(np.abs(diff-np.rint(diff)).max()))
    clock=np.random.default_rng(271119);b1=[]
    for _ in range(131):
        tic=perf_counter();sample(1,clock,clock);b1.append(1000*(perf_counter()-tic))
    assert sum(sizes)==10000 and nfinite==10000 and all(x>0 for x in times)
    data=dict(arm=arm,checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=fitted.digest(path),
        cpu_pool_sha256=fitted.digest(pool),batch=128,nfe=1,sizes=sizes,batch_seconds=times,elapsed_seconds=sum(times),
        median_b1_ms=float(np.median(b1[3:])),b1_ms=b1[3:],warmup_b1=3,warmup_b16=3,finite=nfinite,
        max_chart_difference=dk,max_fractional_difference=df,torch=str(torch.__version__),cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),threads=1,tf32=False,seed=11,composition_seed=22112026,geometry_seed=71831,
        boundary="Chemistry sampling, seed, one model call, host fractional and Cartesian sites and finite check. Array comparison, disk and scientific scoring excluded.")
    fitted.write_json(HERE/"pools"/(arm+"-cuda.json"),data)
    print(arm,data["elapsed_seconds"],data["median_b1_ms"],dk,df,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site",type=Path,nargs="*",default=[])
    args=parser.parse_args();sys.path.extend(map(str,args.site))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    for arm in ("fixed-basis","radial","angular"):run(arm)
