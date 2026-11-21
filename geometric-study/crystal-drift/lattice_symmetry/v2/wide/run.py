"""Matched held-out sensitivity to a fourfold wider traceless cell strain."""
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"csp"))
import model
from lattice_symmetry.v2.csp import train,stage
from lattice_symmetry.v2.physical import Generator
from lattice_symmetry.v2.acceptance import records,objective,change,maxerr

NAME="angular-wide"
SEED=11
STEPS=(16,64,256)


class Wide(Generator):
    def forward(self,cell,frac,z):
        out,pos=super().forward(cell,frac,z)
        a=torch.linalg.solve(cell,out)
        logvol=torch.linalg.slogdet(a).logabsdet
        dev=a*torch.exp(-logvol/3)[:,None,None]
        return out@dev@dev@dev,pos


def make(arm):
    assert arm==NAME
    return Wide(**model.GENERATOR)


def run(seed=SEED):
    torch.manual_seed(431);torch.set_num_threads(1)
    rows=records();rng=np.random.default_rng(911)
    other=[change(c,rng,i)[0] for i,c in enumerate(rows)]
    m=make(NAME).double()
    a=objective(m,rows);b=objective(m,other)
    invariance={k:maxerr(a[k],b[k]) for k in ("descriptor","field","gradient","loss")}
    assert max(invariance.values())<1e-9
    O=torch.eye(3,dtype=torch.float64).repeat(2,1,1)*4
    Z=torch.tensor([[6,8],[8,14]])
    F=torch.rand(2,2,3,dtype=torch.float64)
    narrow=Generator(**model.GENERATOR).double()
    narrow.load_state_dict(m.state_dict())
    lo,fo=narrow(O,F,Z);lw,fw=m(O,F,Z)
    volume=maxerr(torch.linalg.det(lo),torch.linalg.det(lw))
    assert volume<1e-9 and maxerr(fo,fw)==0
    train.new_model=make
    train.geometry=lambda net,arm,z,k,f:model.geometry(net,"angular",z,k,f)
    stage.new_model=make
    stage.service_from=lambda net,arm,prior,device:model.service_from(net,"angular",prior,device)
    train.train((NAME,seed,max(STEPS),"cpu"))
    stage.validate([NAME],[seed],STEPS)
    rng=np.random.default_rng(919511)
    index=rng.integers(128,size=(5000,128))
    result={}
    for u in STEPS:
        files=[model.HERE/"fits"/f"{arm}-{seed}"/f"val-{u:05}.npz" for arm in ("angular",NAME)]
        with np.load(files[0]) as ref,np.load(files[1]) as new:
            assert ref["ids"].tolist()==new["ids"].tolist()
            d=new["energy"]-ref["energy"]
            result[str(u)]=dict(narrow=float(ref["energy"].mean()),wide=float(new["energy"].mean()),
                wide_minus_narrow=float(d.mean()),paired_95=np.quantile(d[index].mean(1),[.025,.975]).tolist(),
                checkpoint_sha256=[str(ref["checkpoint_sha256"]),str(new["checkpoint_sha256"])])
    sa=[model.HERE/"fits"/f"{arm}-{seed}"/"schedule.npz" for arm in ("angular",NAME)]
    with np.load(sa[0]) as a,np.load(sa[1]) as b:
        assert all(np.array_equal(a[k],b[k]) for k in ("anchors","positive"))
    model.write_json(HERE/("result.json" if seed==11 else f"result-{seed}.json"),dict(seed=seed,validation=result,setup_symmetries=invariance,volume_error=volume,
        original_schedule=hashlib.sha256(sa[0].read_bytes()).hexdigest(),criterion_pass=result["256"]["paired_95"][1]<0))
    print(result)


if __name__=="__main__":run()
