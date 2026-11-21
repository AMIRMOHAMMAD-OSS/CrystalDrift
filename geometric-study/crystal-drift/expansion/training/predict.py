"""Generate nested proposals for every public MP-20 target after validation selection."""
import argparse
import json
import os
import time

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"

import numpy as np
import torch

from controls import ROOT,build,generate
from fit import HERE,METHODS,write


@torch.inference_mode()
def pool(method,seed,epoch,cohort,steps):
    tag=("flow"+str(steps) if method=="flow" else method)+f"-seed{seed}"
    folder=HERE/"test"/tag
    folder.mkdir(exist_ok=True,parents=True)
    path=folder/"ready.json"
    if path.exists():
        if json.loads(path.read_text())["pass"]!=epoch:
            raise ValueError("Frozen selection changed")
        return
    state=torch.load(HERE/f"fits/{method}-seed{seed}/pass{epoch}.pt",weights_only=True,map_location="cpu")
    model=build(method).eval()
    model.load_state_dict(state["model"])
    n=len(cohort["ids"])
    kraw=np.lib.format.open_memmap(folder/"k-tmp.npy",mode="w+",dtype="float32",shape=(n,20,6))
    fraw=np.lib.format.open_memmap(folder/"f-tmp.npy",mode="w+",dtype="float32",shape=(n,20,20,3))
    rng=np.random.default_rng(np.random.SeedSequence([20261120,seed]))
    old_rng=np.random.default_rng(100000+seed)
    oldk=old_rng.standard_normal((128*5,6)).astype(np.float32).reshape(128,5,6)
    oldf=old_rng.random((128*5,20,3)).astype(np.float32).reshape(128,5,20,3)
    tic=time.perf_counter()
    for start in range(0,n,16):
        end=min(n,start+16);num=end-start
        z=torch.from_numpy(np.repeat(cohort["z"][start:end],20,axis=0)).long()
        noise_k=rng.normal(size=(num,20,6)).astype(np.float32)
        noise_f=rng.random((num,20,20,3),dtype=np.float32)
        if start<128:
            noise_k[:,:5]=oldk[start:end]
            noise_f[:,:5]=oldf[start:end]
        k,f=generate(model,method,z,torch.from_numpy(noise_k.reshape(-1,6)),torch.from_numpy(noise_f.reshape(-1,20,3)),steps)
        k=(k*state["scale"]+state["mean"]).reshape(num,20,6).numpy()
        f=f.reshape(num,20,20,3).numpy()
        assert np.isfinite(k).all() and np.isfinite(f).all()
        kraw[start:end]=k;fraw[start:end]=f
    kraw.flush();fraw.flush()
    del kraw,fraw
    for field in ("k","f"):
        (folder/(field+"-tmp.npy")).replace(folder/(field+".npy"))
    write(path,dict(model=method,seed=seed,nfe=steps,targets=n,draws=20,
         source_pool="revision/full/mp20/cohort.npz; same ordered targets and noise as the original four compact methods",
         elapsed_seconds=time.perf_counter()-tic,**{"pass":epoch}))
    print(tag,epoch,round(time.perf_counter()-tic,2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(1)
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:
        cohort={k:d[k] for k in ("ids","z")}
    while True:
        selected=HERE/"selection.json"
        choices=json.loads(selected.read_text()) if selected.exists() else {}
        for method in choices:
            for seed in (11,23,37):
                for steps in ((1,16) if method=="flow" else (1,)):
                    pool(method,seed,choices[method]["pass"],cohort,steps)
        if len(choices)==len(METHODS) or not args.watch:break
        time.sleep(4)


if __name__=="__main__":main()
