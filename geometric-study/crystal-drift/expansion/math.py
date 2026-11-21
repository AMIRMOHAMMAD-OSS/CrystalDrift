"""Integrate the actual finite-bank map exactly over all 16 iid two-slot banks."""
import itertools
import json
import sys
from pathlib import Path
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"scaled"),str(ROOT)]
from family import features,within_direction


def main():
    torch.set_num_threads(1)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as f:
        z,k,s=(f[key] for key in ("z","k","f"))
    _,inverse,counts=np.unique(z,axis=0,return_inverse=True,return_counts=True)
    row=np.flatnonzero((counts[inverse]>=2)&(z[:,2]>0)&(z[:,0]!=z[:,2]))[0]
    ids=np.flatnonzero(inverse==inverse[row])[:2]
    kk=(k[ids].astype(float)-k.mean(0))/np.maximum(k.std(0),1e-5)
    atoms=torch.from_numpy(z[ids].astype(np.int64))
    real=features(atoms,torch.tensor(kk),torch.tensor(s[ids],dtype=torch.float64))
    assert real.shape==(2,1306) and (real[0]-real[1]).norm()>0
    q=torch.stack([real[0]+.19*(real[1]-real[0]),real[1]+.25*(real[1]-real[0])])
    result=[];maxswap=0;maxnorm=0
    for offset in [1,.25,.0625,.015625,.00390625,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,0]:
        shift=torch.zeros_like(real);shift[:,0]=offset
        total=torch.zeros_like(q);anchored=torch.zeros_like(q)
        for pi in itertools.product(range(2),repeat=2):
            for ni in itertools.product(range(2),repeat=2):
                p=real[list(pi)][None];n=(real+shift)[list(ni)][None]
                value=within_direction(q[None],p,n)[0]
                total+=value/16
                if pi[0]==0:anchored+=value/8
                swapped=within_direction(q[None],n,p)[0]
                maxswap=max(maxswap,float((swapped+value).abs().max()))
                maxnorm=max(maxnorm,float(value.norm(dim=-1).max()))
        result.append(dict(offset=offset,max_expected_query_norm=float(total.norm(dim=-1).max()),
                           fixed_first_positive_norm=float(anchored.norm(dim=-1).max())))
    assert result[-1]["max_expected_query_norm"]<1e-12 and maxswap<1e-12
    assert result[-1]["fixed_first_positive_norm"]>1e-3 and maxnorm<=np.sqrt(2)+1e-12
    out=dict(reference_indices=ids.tolist(),finite_bank_pairs=16,dimensions=1306,queries=2,temperatures=[.3,.8],
             rms_epsilon=1e-8,exchange_sign_error=maxswap,maximum_observed_query_norm=maxnorm,series=result)
    (HERE/"weak-limit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(result)


if __name__=="__main__":main()
