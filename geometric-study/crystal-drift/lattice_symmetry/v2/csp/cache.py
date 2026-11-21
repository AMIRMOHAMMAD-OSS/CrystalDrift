"""Cache all train-only positive descriptors and fit common scaling to the serialized vectors."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
import time
import numpy as np
import torch
from model import DESCRIPTOR,HERE,ROOT,digest,make_cell,save_npz,write_json
from lattice_symmetry.v2.physical import blocks,features


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    p=ROOT/"scaled/mp20-data/train.npz";data=np.load(p)
    z,k,f=[data[x] for x in ("z","k","f")]
    slices=blocks()
    d=max(v.stop for v in slices.values())
    cache=np.lib.format.open_memmap(HERE/"cache.building",mode="w+",dtype="float32",shape=(len(z),d))
    tic=time.perf_counter()
    for lo in range(0,len(z),48):
        ix=slice(lo,lo+48)
        val=features(make_cell(torch.tensor(k[ix]).float()),torch.tensor(f[ix]).float(),torch.tensor(z[ix]).long(),**DESCRIPTOR)
        cache[ix]=val.numpy()
    elapsed=time.perf_counter()-tic
    scale=np.ones(d,np.float32)
    mean=cache.mean(0,dtype=np.float64)
    vals={}
    for key,sl in slices.items():
        ss=0.
        for lo in range(0,len(z),256):ss+=np.square(cache[lo:lo+256,sl]-mean[sl]).sum()
        vals[key]=max(.01,(ss/len(z))**.5)
        scale[sl]=vals[key]
    assert np.isfinite(cache).all() and np.isfinite(scale).all()
    cache.flush();del cache
    (HERE/"cache.building").replace(HERE/"positive.npy")
    save_npz(HERE/"scaling.npz",mean=mean.astype(np.float32),scale=scale)
    write_json(HERE/"cache.json",dict(train_records=len(z),dimensions=d,denominator=vals,seconds=elapsed,threads=1,
        split_sha256=digest(p),feature_source_sha256=digest(HERE.parent/"physical.py"),cache_sha256=digest(HERE/"positive.npy")))
    print("Cached",len(z),"cells in",elapsed,"seconds",flush=True)

if __name__=="__main__":main()
