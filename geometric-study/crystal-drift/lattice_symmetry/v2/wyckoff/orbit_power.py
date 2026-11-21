"""An optional label-invariant second moment of per-orbit angular environments."""
import numpy as np
import torch
from charts import chart
from labels import ROOT
import sys
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v2.neighbors import edges
from lattice_symmetry.v2.physical import harmonics

def orbit_ids(group,template):
    lengths=[len(chart(group,l)[0]) for _,l in template]
    return np.repeat(np.arange(len(lengths)),lengths)

def variance(l,f,z,ids):
    """Average the within-(species,multiplicity) variance of orbit powers, over occupied orbits."""
    B,N=z.shape; s,d,b,v=edges(l,f,z,4.)
    q=v/4.;x=q.square().sum(-1)
    w=(1-x).clamp_min(0).pow(3)
    center=[]
    for y in harmonics(q):
        co=l.new_zeros(B*N,y.size(1)).index_add(0,s,w[:,None]*y)
        center.append((co*co).sum(1))
    power=torch.stack(center,-1).reshape(B,N,3)
    result=[]
    for i,index in enumerate(ids):
        index=torch.as_tensor(index,dtype=torch.long,device=z.device)
        occup,counts=torch.unique(index,sorted=True,return_counts=True)
        L=len(index)
        # Relabel arbitrary site-orbit identifiers to dense slots.
        _,dense=torch.unique(index,sorted=True,return_inverse=True)
        orbital=l.new_zeros(len(occup),3).index_add_(0,dense,power[i,:L])/counts[:,None]
        elements=torch.zeros(len(occup),dtype=z.dtype,device=z.device).scatter_(0,dense,z[i,:L])
        unique,cl,num=torch.unique(torch.stack([elements,counts],1),dim=0,return_inverse=True,return_counts=True)
        means=l.new_zeros(len(unique),3).index_add_(0,cl,orbital)/num[:,None]
        result.append(((orbital-means[cl])**2).mean(0))
    return torch.stack(result)
