"""Verify common scheduling, IID feature forces and the one-step control's JVP numerically."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from controls import ROOT,MeanFlow
from fit import HERE,write
from family import within_direction


def check_jvp():
    torch.manual_seed(20261120)
    net=MeanFlow().double()
    with torch.no_grad():net.interval.copy_(torch.linspace(-.16,.12,6))
    z=torch.tensor([[8,8,14,14,0],[11,17,0,0,0]])
    k=torch.randn(2,6,dtype=torch.float64)
    f=torch.rand(2,5,3,dtype=torch.float64)
    r=torch.tensor([[.2],[.3]],dtype=torch.float64)
    t=torch.tensor([[.7],[.8]],dtype=torch.float64)
    dk=torch.randn_like(k);df=torch.randn_like(f)
    def fn(k,f,t):return net.velocity(z,k,f,r,t)
    out,j=torch.func.jvp(fn,(k,f,t),(dk,df,torch.ones_like(t)))
    h=1e-5
    plus=fn(k+h*dk,f+h*df,t+h);minus=fn(k-h*dk,f-h*df,t-h)
    errors=[float((jj-(a-b)/(2*h)).abs().max().detach()) for jj,a,b in zip(j,plus,minus)]
    assert max(errors)<2e-7
    q=torch.randn((3,4,17),dtype=torch.float64);p=torch.randn_like(q);n=torch.randn_like(q)
    swap=float((within_direction(q,p,n)+within_direction(q,n,p)).abs().max())
    assert swap<1e-12
    return dict(central_difference_epsilon=h,jvp_lattice_max_abs_error=errors[0],jvp_site_max_abs_error=errors[1],
                independent_bank_swap_max_abs_error=swap)


def main():
    data=check_jvp();hashes={}
    for epoch in (1,3,10):
        for seed in (11,23,37):
            roots=[HERE/f"fits/{m}-seed{seed}/pass{epoch}.json" for m in ("drift","iid","flow","meanflow")]
            if not all(p.exists() for p in roots):continue
            fits=[json.loads(p.read_text()) for p in roots]
            assert len(set(x["schedule_sha256"] for x in fits))==1
            assert fits[0]["bank_sha256"]==fits[2]["bank_sha256"]==fits[3]["bank_sha256"]
            assert fits[0]["bank_sha256"]!=fits[1]["bank_sha256"]
            hashes[f"{epoch}/{seed}"]=fits[0]["schedule_sha256"]
    data["paired_schedule_sha256"]=hashes
    write(HERE/"training-checks.json",data)
    print(json.dumps(data,indent=1))


if __name__=="__main__":main()
