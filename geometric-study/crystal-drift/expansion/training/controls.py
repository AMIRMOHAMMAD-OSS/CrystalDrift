"""Endpoint average velocity on a periodic motif and standardized log lattice."""
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scaled"))
from family import SetNetwork, draw, mean_sites


class MeanFlow(SetNetwork):
    def __init__(self):
        super().__init__()
        # Six learned offsets convey the interval length through the existing
        # global context. The instantaneous-time embedding remains unchanged.
        self.interval = torch.nn.Parameter(torch.zeros(6))

    def velocity(self, z, k, f, r, t):
        return self(z, k + (t-r)*self.interval, f, t)


def meanflow_loss(net,z,clean_k,clean_f):
    noise_k=torch.randn_like(clean_k)
    noise_f=torch.rand_like(clean_f)
    mask=(z>0).unsqueeze(-1)
    order=(z.masked_fill(z==0,120).float()+torch.rand_like(z,dtype=clean_k.dtype)*.5).argsort(1)
    f=clean_f.gather(1,order[:,:,None].expand_as(clean_f))
    df=(noise_f-f+.5).remainder(1)-.5
    df=(df-mean_sites(df,mask)[:,None])*mask
    dk=noise_k-clean_k
    times=torch.rand((len(z),2),device=z.device)
    t=times.max(1,keepdim=True).values
    r=times.min(1,keepdim=True).values
    r=torch.where(torch.rand_like(r)<.25,t,r)
    kt=noise_k+(t-1)*dk
    ft=(noise_f+(t-1)[:,:,None]*df).remainder(1)
    def average(kk,ff,tt):
        return net.velocity(z,kk,ff,r,tt)
    u,du=torch.func.jvp(average,(kt,ft,t),(dk,df,torch.ones_like(t)))
    goal_k=(dk-(t-r)*du[0]).detach()
    goal_f=(math.sqrt(12)*df-(t-r)[:,:,None]*du[1]).detach()
    return (u[0]-goal_k).square().mean()+((u[1]-goal_f).square()*mask).sum()/(3*mask.sum())


def build(method):
    return MeanFlow() if method=="meanflow" else SetNetwork()


@torch.inference_mode()
def generate(net,method,z,k,f,steps=1):
    if method=="meanflow":
        zero=k.new_zeros(len(k),1)
        uk,uf=net.velocity(z,k,f,zero,zero+1)
        return k-uk,(f-uf/math.sqrt(12)).remainder(1)*(z>0).unsqueeze(-1)
    return draw(net,"flow" if method=="flow" else "drift",z,k,f,steps)
