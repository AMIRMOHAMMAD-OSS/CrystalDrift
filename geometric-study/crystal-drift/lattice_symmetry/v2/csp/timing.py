"""Full GPU timing replay using the archived 128-ID, cap-20 serial protocol."""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[v]="1"
import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import numpy as np
import torch
from model import ARMS,HERE,ROOT,checkpoint_path,digest,load_service,write_json

sys.path.insert(0,str(ROOT/"expansion/training"))
import gpu as protocol

BASE={"old-drift":"drift","old-flow16":"flow16","meanflow":"meanflow","uflow":"uflow","retrieval":"retrieval"}
ORIGINAL=protocol.service


def timed_service(method,z,weights,temp):
    if method in BASE:return ORIGINAL(BASE[method],z,weights,temp)
    propose=load_service(method,11,"cuda")
    def call(z,rng):
        k,f=propose(z,rng)
        f=np.where(z[:,:,None]>0,np.remainder(f,1),0.).astype(np.float64)
        cell,f,cart=protocol.decode(k,f)
        return z.copy(),cell,f,cart,k
    return call


def main():
    p=argparse.ArgumentParser();p.add_argument("--weights",type=Path)
    p.add_argument("--site",nargs="*",type=Path,default=[])
    p.add_argument("--methods",nargs="+",choices=(*BASE,*ARMS,"prior"),default=(*BASE,*ARMS,"prior"))
    p.add_argument("--batch",type=int,default=24)
    args=p.parse_args();sys.path.extend(map(str,args.site))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    protocol.GPU=HERE/"gpu";protocol.GPU.mkdir(exist_ok=True)
    protocol.service=timed_service
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as fp:ids,z=fp["ids"][:128],fp["z"][:128]
    with tempfile.TemporaryDirectory() as temp:
        for method in args.methods:
            out=protocol.GPU/f"{method}-settings.json"
            if out.exists():continue
            torch.cuda.reset_peak_memory_stats()
            start=time.time();load_start=os.getloadavg()
            protocol.measure(method,ids,z,args,Path(temp))
            end=time.time();load_end=os.getloadavg()
            cfg=dict(method=method,checkpoint_sha256=None,torch=torch.__version__,architecture="prior-only" if method=="prior" else "local" if method in ARMS else BASE[method],
                     device=torch.cuda.get_device_name(),cuda=torch.version.cuda,platform=platform.platform(),threads=1,tf32=False,
                     output="CPU species, float64 canonical lattice, wrapped fractional and Cartesian coordinates",
                     warmup_calls=5,k_max=20,batch_timing=24,requests=10000,software_protocol_sha256=digest(ROOT/"expansion/training/gpu.py"),
                     seed=11,max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     timing_process_start_unix=start,timing_process_end_unix=end,host_processors=os.cpu_count(),
                     linux_load_1_5_15_min_start=load_start,linux_load_1_5_15_min_end=load_end)
            if method=="prior":cfg["fitted_prior_sha256"]=digest(checkpoint_path(method,11))
            elif method in ARMS:cfg["checkpoint_sha256"]=digest(checkpoint_path(method,11))
            elif method=="uflow":cfg["checkpoint_sha256"]=digest(args.weights)
            elif method=="retrieval":cfg["train_sha256"]=digest(ROOT/"scaled/mp20-data/train.npz")
            else:
                key="flow" if method=="old-flow16" else BASE[method]
                pas=json.loads((ROOT/"expansion/training/selection.json").read_text())[key]["pass"]
                cfg["checkpoint_sha256"]=digest(ROOT/f"expansion/training/fits/{key}-seed11/pass{pas}.pt")
            for name in ("episodes","10000"):cfg[name+"_sha256"]=digest(protocol.GPU/f"{method}-{name}.npz")
            write_json(out,cfg);torch.cuda.empty_cache()

if __name__=="__main__":main()
