"""Matched set networks for a one-call drift generator and conditional flow matching."""
import math

import torch
from torch import nn


TAU=2*math.pi
WAVES=((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),
       (1,-1,0),(1,0,-1),(0,1,-1),(2,0,0),(0,2,0),(0,0,2),(1,1,1))


def species_averages(z,dtype):
    valid=(z>0).unsqueeze(-1)
    same=(z[:,:,None]==z[:,None,:]) & valid & valid.transpose(1,2)
    counts=same.sum(-1,keepdim=True)
    return valid.to(dtype),same.to(dtype)/counts.clamp_min(1),counts.to(dtype)


def mean_sites(x,mask):
    return (x*mask).sum(1)/mask.sum(1).clamp_min(1)


class SetNetwork(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.embed=nn.Embedding(119,16,padding_idx=0)
        self.site=nn.Sequential(nn.Linear(56,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.pool=nn.Sequential(nn.Linear(width+26,width),nn.SiLU())
        self.local=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,3))
        self.cell=nn.Linear(width,6)

    def forward(self,z,k,f,t):
        mask,same,counts=species_averages(z,k.dtype)
        e=self.embed(z); n=mask.sum(1)
        context=torch.cat([mean_sites(e,mask),k,t,t.sin(),t.cos(),n/20],-1)
        c,s=(TAU*f).cos(),(TAU*f).sin()
        all_c,all_s=mean_sites(c,mask)[:,None,:],mean_sites(s,mask)[:,None,:]
        own_c,own_s=same@c,same@s
        phase=torch.cat([c*all_c+s*all_s,s*all_c-c*all_s,
                         c*own_c+s*own_s,s*own_c-c*own_s],-1)
        sites=self.site(torch.cat([e,phase,counts/n[:,None,:],mask,
                                    context[:,None,:].expand(-1,z.shape[1],-1)],-1))
        pooled=self.pool(torch.cat([mean_sites(sites,mask),context],-1))
        shifts=self.local(torch.cat([sites,pooled[:,None,:].expand_as(sites)],-1))*mask
        return self.cell(pooled),shifts


def drift_draw(net,z,k,f):
    cell,shift=net(z,k,f,k.new_zeros((len(k),1)))
    return cell+.15*k,(f+.5*shift.tanh()).remainder(1)*(z>0).unsqueeze(-1)


def features(z,k,f):
    """Species-resolved autocorrelations and cross-correlations to two anchors.

    Each site's channels describe its entire chemical species. Sorted Z order
    aligns these repeated channels within an exact-composition training group.
    """
    mask,same,_=species_averages(z,k.dtype)
    phase=TAU*(f@f.new_tensor(WAVES).T)
    c,s=phase.cos(),phase.sin()
    own_c,own_s=same@c,same@s
    sum_c,sum_s=mean_sites(c,mask)[:,None,:],mean_sites(s,mask)[:,None,:]
    first_c,first_s=own_c[:,:1],own_s[:,:1]
    feat=torch.cat([own_c.square()+own_s.square(),own_c*sum_c+own_s*sum_s,
                    own_s*sum_c-own_c*sum_s,own_c*first_c+own_s*first_s,
                    own_s*first_c-own_c*first_s],-1)
    feat=feat*mask/(mask.sum(1,keepdim=True)*len(WAVES)*5).sqrt()
    return torch.cat((k/math.sqrt(6),feat.flatten(1)),-1)


@torch.no_grad()
def within_direction(q,p,neg):
    ref=torch.cat([p,neg],1)
    distances=torch.cdist(q,ref)
    result=torch.zeros_like(q)
    for tau in (.3,.8):
        logits=-distances/tau
        a=(.5*(logits.log_softmax(1)+logits.log_softmax(2))).exp()
        wp,wn=a.chunk(2,-1)
        drift=(wp@p)*wn.sum(-1,keepdim=True)-(wn@neg)*wp.sum(-1,keepdim=True)
        result+=drift/(drift.norm(dim=-1).square().mean()+1e-8).sqrt()
    return result/2


def objective(net,method,z,k1,f1,group_size):
    k0=torch.randn_like(k1); f0=torch.rand_like(f1); mask=(z>0).unsqueeze(-1)
    if method=='flow':
        # Equal-species permutations and quotienting the common translation
        # prevent arbitrary reference atom labels/origins becoming targets.
        keys=z.masked_fill(z==0,120).float()+torch.rand_like(z,dtype=k1.dtype)*.5
        perm=keys.argsort(1)
        f1=f1.gather(1,perm[:,:,None].expand_as(f1))
        df=(f1-f0+.5).remainder(1)-.5
        df=(df-mean_sites(df,mask)[:,None])*mask
        t=torch.rand((len(z),1),device=z.device)
        dk,dx=net(z,k0+t*(k1-k0),(f0+t[:,:,None]*df).remainder(1),t)
        return (dk-(k1-k0)).square().mean()+((dx-math.sqrt(12)*df).square()*mask).sum()/(3*mask.sum())
    with torch.no_grad():
        kr,fr=drift_draw(net,z,torch.randn_like(k0),torch.rand_like(f0))
        negative,positive=features(z,kr,fr),features(z,k1,f1)
    kp,fp=drift_draw(net,z,k0,f0)
    q=features(z,kp,fp)
    shape=(-1,group_size,q.shape[-1])
    d=within_direction(q.detach().reshape(shape),positive.reshape(shape),negative.reshape(shape))
    return .5*(q-q.detach()-d.reshape_as(q)).square().sum(-1).mean()


@torch.inference_mode()
def draw(net,method,z,k,f,steps):
    if method=='drift':return drift_draw(net,z,k,f)
    mask=(z>0).unsqueeze(-1)
    for j in range(steps):
        t=k.new_full((len(k),1),j/steps)
        dk,df=net(z,k,f,t)
        k=k+dk/steps; f=(f+df/(math.sqrt(12)*steps)).remainder(1)*mask
    return k,f
