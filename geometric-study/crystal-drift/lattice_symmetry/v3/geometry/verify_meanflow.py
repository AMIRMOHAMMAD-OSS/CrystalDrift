"""Check sign, stopped JVP target and trainable gradient of the same-backbone objective."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ..models.network import OrbitNet
from ..models.train import meanflow


def main():
    here=Path(__file__).resolve().parent
    torch.set_num_threads(1);torch.manual_seed(19366)
    with np.load(here.parent/"models/cache/train-000.npz") as d:
        x=[torch.as_tensor(d[f"x{i}"][:6],dtype=torch.long if i in (2,3) else torch.float64)
           for i in range(8)]
        g=[torch.from_numpy(d[k][:6]).double() for k in ("targetL","targetF")]
    net=OrbitNet(width=32,depth=1).double()
    seed=225901
    actual,info=meanflow(net,x,*g,np.random.default_rng(seed))
    grads=torch.cat([v.flatten() for v in torch.autograd.grad(actual,net.parameters())])
    L,F,Z,O,m,free,J,P=x
    ut=np.random.default_rng(seed).random((3,len(L)))
    t=torch.from_numpy(np.maximum(ut[0],ut[1]));r=torch.from_numpy(np.where(ut[2]<.3,t,np.minimum(ut[0],ut[1])))
    def evaluate(time):
        a=L+(1-time[:,None,None])*g[0]
        f=F+(1-time[:,None,None])*g[1]
        return net(a,f,Z,O,m,free,J,P,r,time,residual=True)
    tau=1e-6
    with torch.no_grad():
        forward=evaluate(t+tau);back=evaluate(t-tau)
        derivative=[(a-b)/(2*tau) for a,b in zip(forward,back)]
        target=[a-(t-r)[:,None,None]*b for a,b in zip(g,derivative)]
    pL,pF=evaluate(t)
    control=.5*(((pL-target[0])/4).square().mean()+
        (((pF-target[1])@L)/4).square().sum((1,2)).div(3*(Z>0).sum(1)).mean())
    ga=torch.cat([v.flatten() for v in torch.autograd.grad(control,net.parameters())])
    err=abs(float(actual.detach()-control.detach()));gerr=float((ga-grads).norm()/grads.norm())
    assert err<2e-8 and gerr<2e-7,(err,gerr)
    result=dict(seed=seed,finite_difference=tau,instances=len(L),torch=torch.__version__,
                actual=float(actual.detach()),finite_difference_loss=float(control.detach()),
                loss_absolute_difference=err,relative_gradient_error=gerr,
                note="Independent centered derivative along x_t=seed+(1-t)Delta. The output is a backward displacement-rate. The coordinate alignment and componentwise lattice loss use the canonical training basis.")
    (here/"meanflow-objective-test.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(err,gerr)

if __name__=="__main__":main()
