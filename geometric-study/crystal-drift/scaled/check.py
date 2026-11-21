"""Independent numerical checks of multielement conditioning, periodicity and scoring."""
import json
from itertools import product

import numpy as np
import torch

from evaluate import physical
from family import SetNetwork,drift_draw,features,objective
from prepare_mp20 import pack_gram


def main():
    torch.manual_seed(19);torch.set_num_threads(1)
    z=torch.tensor([[8,8,12,13,27,49,0],[3,11,11,17,17,53,0]])
    f=torch.rand(2,7,3);k=torch.randn(2,6);model=SetNetwork().eval()
    shift=torch.rand(2,1,3);perm=torch.randperm(7);mask=(z>0).unsqueeze(-1)
    with torch.no_grad():
        a,b=drift_draw(model,z,k,f);a2,b2=drift_draw(model,z,k,(f+shift).remainder(1))
        trans_error=(((b2-b-shift+.5).remainder(1)-.5)*mask).abs().max().item()
        assert trans_error<1e-5 and torch.allclose(a,a2,atol=1e-6)
        ap,bp=drift_draw(model,z[:,perm],k,f[:,perm]);assert torch.allclose(a,ap,atol=1e-6)
        assert torch.allclose(b[:,perm],bp,atol=1e-6)
        f2=features(z,k,f);assert torch.allclose(f2,features(z,k,(f+shift).remainder(1)),atol=1e-6)
        z2=z.clone();z2[0,2]=9
        assert not torch.allclose(a,drift_draw(model,z2,k,f)[0])
    for method in ('drift','flow'):
        zg=z.repeat_interleave(4,0); kg=k.repeat_interleave(4,0); fg=f.repeat_interleave(4,0)
        model.zero_grad();loss=objective(model,method,zg,kg,fg,4);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
    rng=np.random.default_rng(201); largest=0
    for _ in range(10):
        cell=np.diag([2.5,3.,4.])+rng.normal(scale=.3,size=(3,3));g=cell@cell.T;kp=pack_gram(g)
        x=rng.random((7,3)); dis=physical(kp,x,np.ones(7))
        shifts=np.array(list(product(range(-3,4),repeat=3)));difference=x[:,None,None]-x[None,:,None]+shifts
        d=np.linalg.norm(difference@cell,axis=-1);d[np.arange(7),np.arange(7),np.where(~shifts.any(1))[0][0]]=np.inf
        near=d.min(axis=(1,2)); known=np.r_[np.log(abs(np.linalg.det(cell))/7),np.min(near),np.median(near),np.quantile(near,.9)]
        largest=max(largest,float(np.max(abs(dis-known))))
    assert largest<1e-7
    print(json.dumps({'translation_error':trans_error,'nearest_image_feature_max_error':largest,
                      'arbitrary_site_permutation':True,'interior_element_conditioning':True,'two_finite_objective_gradients':True}))


if __name__=='__main__':main()
