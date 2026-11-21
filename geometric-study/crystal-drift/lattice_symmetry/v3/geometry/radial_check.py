"""Verify the late radial-resolution sensitivity against both the release and independent oracles."""
import json
from pathlib import Path
import numpy as np
import torch
from .descriptors import dense_pair_shells,feature_blocks,fit_stats,combine,bank_direction
from .verify import pack,fixtures,change
from lattice_symmetry.v2.physical import features,blocks
from lattice_symmetry.v2.acceptance import independent_numpy

def main():
    here=Path(__file__).resolve().parent
    rng=np.random.default_rng(617209);rr=fixtures()
    reps=[change(r,rng,i)[0] for i,r in enumerate(rr)]
    torch.set_num_threads(1);results={}
    for dt in (torch.float64,torch.float32):
        a,b=pack(rr,dt,"cpu"),pack(reps,dt,"cpu")
        val=feature_blocks(*a,arm="radial4");other=feature_blocks(*b,arm="radial4")
        sl=blocks(4,7)["radial"];native=features(*a[:3],angular=False,n_radial=4)[:,sl]
        x=val["radial4"];err=lambda u,v:float((u-v).abs().max())
        # Independent NumPy pair accumulator uses each center's directly enumerated images.
        control=np.stack([independent_numpy((r[0],r[1],r[2]),cutoff=4.,K=7,R=4)[sl] for r in rr])
        stats=fit_stats(val)
        y=combine(val,stats).reshape(3,3,-1);yp=combine(other,stats).reshape_as(y)
        vec=lambda z:bank_direction(z[0:1],z[1:2],z[2:3])
        result=dict(v2_radial4_parity=err(x,native),independent_max_abs=float(np.abs(x.numpy()-control).max()),
             representation=err(x,other["radial4"]),field_representation=err(vec(y),vec(yp)),dimensions=x.shape[1])
        assert max(v for k,v in result.items() if k!="dimensions") < (1e-9 if dt==torch.float64 else .0002)
        results[str(dt)]=result
    (here/"radial4-test.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8");print(results)

if __name__=="__main__":main()
