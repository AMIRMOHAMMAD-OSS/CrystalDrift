"""Generate ordered raw test draws with resumable per-target shards and an ID/key crosswalk."""
import os
for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[variable]="1"
import argparse
import json
import time
import numpy as np
import torch
from model import ARMS,HERE,ROOT,SEEDS,checkpoint_path,digest,load_service,save_npz,write_json


def predict(arm,seed,device="cpu"):
    ck=checkpoint_path(arm,seed)
    folder=HERE/"test"/f"{arm}-seed{seed}";folder.mkdir(exist_ok=True,parents=True)
    ready=folder/"ready.json";done=folder/"progress.json"
    if ready.exists():
        assert json.loads(ready.read_text())["checkpoint_sha256"]==digest(ck);return
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as ref:ids=ref["ids"];z=ref["z"]
    with np.load(ROOT/"scaled/mp20-data/test.npz") as raw:assert set(raw["ids"])==set(ids) and len(set(ids))==9046
    fun=load_service(arm,seed,device)
    sha=digest(ck)
    old=json.loads(done.read_text()) if done.exists() else None
    if old:
        assert old["checkpoint_sha256"]==sha
        start=old["next_target"];elapsed=old["elapsed"]
    else:start=0;elapsed=0.
    kn=folder/("k.npy" if (folder/"k.npy").exists() else "k-pending.npy")
    fn=folder/("f.npy" if (folder/"f.npy").exists() else "f-pending.npy")
    def array(p,shape):
        if start:return np.load(p,mmap_mode="r+")
        return np.lib.format.open_memmap(p,mode="w+",dtype=np.float32,shape=shape)
    kk=array(kn,(len(ids),20,6));ff=array(fn,(len(ids),20,20,3))
    tic=time.perf_counter()
    for i in range(start,len(ids),4):
        j=min(i+4,len(ids));n=j-i
        rng=np.random.default_rng(np.random.SeedSequence([410932,seed,i]))
        k,f=fun(np.repeat(z[i:j],20,axis=0),rng)
        if not np.isfinite(k).all() or not np.isfinite(f).all():raise FloatingPointError(f"Nonfinite output at {i}")
        kk[i:j]=k.reshape(n,20,6);ff[i:j]=f.reshape(n,20,20,3)
        if j%256==0 or j==len(ids):
            kk.flush();ff.flush()
            write_json(done,dict(next_target=j,checkpoint_sha256=sha,elapsed=elapsed+time.perf_counter()-tic))
    del kk,ff
    if kn.name!="k.npy":kn.replace(folder/"k.npy")
    if fn.name!="f.npy":fn.replace(folder/"f.npy")
    save_npz(folder/"crosswalk.npz",ids=ids,z=z)
    write_json(ready,dict(checkpoint_sha256=sha,selection_sha256=sha if arm=="prior" else digest(HERE/"selection.json"),
            targets=len(ids),draws=20,seed=seed,arm=arm,nfe=0 if arm=="prior" else 16 if arm=="flow16" else 1,
            elapsed=elapsed+time.perf_counter()-tic,cell_sha256=digest(folder/"k.npy"),sites_sha256=digest(folder/"f.npy")))
    print(arm,seed,"generated",len(ids)*20,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--arms",nargs="+",default=ARMS)
    p.add_argument("--seeds",type=int,nargs="+",default=SEEDS);p.add_argument("--device",default="cpu")
    a=p.parse_args();torch.set_num_threads(1)
    for s in a.seeds:
        for arm in a.arms:predict(arm,s,a.device)
