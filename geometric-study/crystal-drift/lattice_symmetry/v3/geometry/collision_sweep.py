"""Analytically controlled eight-site homometry and physical-cutoff sweep with the deployed features."""
import csv
import json
import os
from pathlib import Path
for v in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
import numpy as np
import torch
from .descriptors import blocks,feature_blocks

HERE=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    m=np.array([[0,1,4,6],[0,1,3,7]])
    f=torch.zeros((2,8,3));f[:,:,0]=torch.tensor(np.tile(m,(1,2)))/12;f[:,4:,1]=1/3
    z=torch.tensor([[8]*4+[14]*4]*2);o=torch.arange(8).repeat(2,1)
    out=[]
    with torch.inference_mode():
        for a in np.arange(4.,61.,1.):
            l=torch.diag(torch.tensor([a,5.,5.])).repeat(2,1,1)
            common=feature_blocks(l,f,z,o,"angular")["base"]
            row={"a":a}
            for block,sl in blocks().items():row[block]=float((common[0,sl]-common[1,sl]).norm())
            for arm in ("contacts","reciprocal","variance","radial4"):
                feat=feature_blocks(l,f,z,o,arm)[arm]
                row[arm]=float((feat[0]-feat[1]).norm())
            out.append(row)
    long=[r for r in out if r["a"]>=48.]
    assert max(abs(v) for r in long for n,v in r.items() if n!="a")<1e-11
    assert max(r["reciprocal"] for r in out)<1e-12
    assert max(r["power0"] for r in out)>.01
    with (HERE/"decorated-cutoff-sweep.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print("Long-cell maximum",max(abs(v) for r in long for n,v in r.items() if n!="a"),
          "overall maxima",{k:max(r[k] for r in out) for k in out[0] if k!="a"})
if __name__=="__main__":main()
