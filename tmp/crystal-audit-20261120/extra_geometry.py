from pathlib import Path
import sys,json
import torch
from math import sqrt,pi
base=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(base))
from model import crystal_features,within_formula_field
from scaled.family import features as scaled_features,SetNetwork
from model import CrystalGenerator
torch.set_default_dtype(torch.float64)
torch.manual_seed(116)
sets=((0,1,4,6),(0,1,3,7))
z=torch.tensor([[8]*4+[14]*4+[0]*12]*2)
k=torch.zeros((2,6))
f=torch.zeros((2,20,3))
for i,s in enumerate(sets):
    f[i,:4,0]=f[i,4:8,0]=torch.tensor(s)/12
    f[i,4:8,1]=1/3
v=scaled_features(z,k,f);b=crystal_features(k,f[:,:8],z[:,:8],'multi')
def B(f):
    a=torch.exp(2j*pi*f[:4,0]).mean()
    bb=torch.exp(4j*pi*f[4:8,0]).mean()
    return a*a*bb.conj()
q,p,n=[torch.randn(3,4,70) for _ in range(3)]
V=within_formula_field(q,p,n)
sw=(V+within_formula_field(q,n,p)).abs().max().item()
theta=torch.randn(70,requires_grad=True)
live=(q*theta)
with torch.no_grad(): target=live+within_formula_field(live,p,n)
loss=.5*(live-target).square().sum(-1).mean()
g=torch.autograd.grad(loss,theta)[0]
back=(g+(q*(target-live.detach())).mean((0,1))).abs().max().item()
result=dict(binary_max_abs=float((b[0]-b[1]).abs().max()),multielement_max_abs=float((v[0]-v[1]).abs().max()),multielement_size=v.shape[-1],collinear=[(float(B(x).real),float(B(x).imag)) for x in f],within_block_swap_error=sw,stop_gradient_error=back,param_counts=[sum(p.numel() for p in net.parameters()) for net in (CrystalGenerator(),SetNetwork())])
print(json.dumps(result))

