"""Run official one-call full-size uFlowCSP on the same 10,000 train-sampled compositions."""
import json
import sys
from pathlib import Path
from time import perf_counter
import torch
import numpy as np
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.uflow import runtime
from lattice_symmetry.v3.evaluation.predict import atomic_npz,h
from data import pack_gram

@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.manual_seed(813727)
    net=runtime.model("released","cuda").eval();flow=runtime.transport()
    with np.load(HERE/"pools/prior.npz") as ref:
        ids=ref["ids"];zz=np.sort(ref["z"],axis=1)[:,::-1].copy()
        source=ref["chemistry_source"]
    # Restore canonical increasing positive atomic number before padding.
    z=np.zeros_like(zz)
    for i,row in enumerate(zz):z[i,:np.count_nonzero(row)]=np.sort(row[row>0])
    f=np.zeros((*z.shape,3),np.float32);k=np.full((len(z),6),np.nan);good=np.zeros(len(z),bool)
    start=perf_counter()
    for i in range(0,len(z),32):
        L,F,Z=runtime.propose(net,flow,z[i:i+32])
        assert np.array_equal(Z,z[i:i+len(Z)])
        f[i:i+len(Z)]=F
        for j,l in enumerate(L,i):
            if abs(np.linalg.det(l))>.1 and np.isfinite(l).all() and np.isfinite(f[j]).all():
                k[j]=pack_gram(l.astype(float)@l.astype(float).T);good[j]=True
    elapsed=perf_counter()-start
    dst=HERE/"pools/released-uflow.npz"
    atomic_npz(dst,ids=ids,z=z,k=k,f=f,valid_output=good,chemistry_source=source)
    dst.with_suffix(".json").write_text(json.dumps(dict(draws=len(z),accepted=int(good.sum()),seconds=elapsed,
        cpu_threads=1,batch=32,device=torch.cuda.get_device_name(),seed=813727,sha=h(dst),scope="geometry proposals on common, presampled empirical formulas")))
if __name__=="__main__":main()
