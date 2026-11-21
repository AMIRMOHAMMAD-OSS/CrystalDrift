"""Numerically audit both shipped representations, the new channel, and bank normalization."""
import json
import math

import numpy as np
import torch

from run import HERE, STUDY, fingerprint, model
from scaled import family


def error(a,b):
    return float((a-b).abs().max())


def geometry_checks():
    torch.manual_seed(116)
    torch.set_default_dtype(torch.float64)
    z=torch.tensor([[8]*4+[14]*4]*2)
    f=torch.zeros(2,8,3)
    for i,integers in enumerate(((0,1,4,6),(0,1,3,7))):
        f[i,:,0]=torch.tensor(integers).repeat(2)/12
    f[:,4:,1]=1/3
    k=torch.zeros(2,6)
    z20=torch.nn.functional.pad(z,(0,12))
    f20=torch.nn.functional.pad(f,(0,0,0,12))
    cases={}
    for arm in ("pair","original","harmonic"):
        v=fingerprint(k,f,z,arm)
        cases[arm]=dict(dimension=v.shape[-1],max_absolute_difference=error(v[0],v[1]))
    s=family.features(z20,k,f20)
    cases["scaled"]=dict(dimension=s.shape[-1],max_absolute_difference=error(s[0],s[1]))
    assert cases["original"]["max_absolute_difference"]<1e-14 and cases["scaled"]["max_absolute_difference"]<1e-14
    assert abs(cases["harmonic"]["max_absolute_difference"]-1/(16*math.sqrt(2)))<1e-14
    # Three, four, and one species in sorted slots. The final row covers one-site cells.
    z=torch.zeros(3,20,dtype=torch.long)
    z[0,:6]=torch.tensor([6,6,8,8,22,22])
    z[1,:7]=torch.tensor([6,8,8,16,16,26,26]);z[2,0]=29
    f=torch.rand(3,20,3);k=torch.randn(3,6)
    perm=torch.arange(20).expand(3,-1).clone()
    perm[0,:6]=torch.tensor([1,0,3,2,5,4]);perm[1,:7]=torch.tensor([0,2,1,4,3,6,5])
    shift=torch.rand(3,1,3)
    altered=(f.gather(1,perm[...,None].expand_as(f))+shift).remainder(1)
    net=family.SetNetwork().eval()
    c0,f0=family.drift_draw(net,z,k,f);c1,f1=family.drift_draw(net,z,k,altered)
    residue=(f1-f0.gather(1,perm[...,None].expand_as(f))-shift+.5).remainder(1)-.5
    v0=family.features(z,k,f);v1=family.features(z,k,altered)
    # Compare full features against an independent species-level complex expression.
    amplitudes=[]
    waves=f.new_tensor(family.WAVES)
    for row in range(len(z)):
        active=z[row]>0
        ph=torch.exp(2j*math.pi*(f[row,active]@waves.T))
        species=z[row,active]
        lowest=ph[species==species[0]].mean(0)
        all_sites=ph.mean(0)
        features=torch.zeros(20,65)
        for j,label in enumerate(species):
            a=ph[species==label].mean(0)
            g=a*all_sites.conj();b=a*lowest.conj()
            features[j]=torch.cat((a.abs().square(),g.real,g.imag,b.real,b.imag))
        amplitudes.append(torch.cat((k[row]/math.sqrt(6),features.flatten()/math.sqrt(65*len(species)))))
    symmetries=dict(scaled_generator_cell=error(c0,c1),scaled_generator_sites=float(residue[z>0].abs().max()),
                    scaled_descriptor=error(v0,v1),scaled_formula_vs_independent=error(v0,torch.stack(amplitudes)))
    q,p,n=[torch.randn(3,4,19) for _ in range(3)]
    r=model.within_formula_field(q,p,n)
    second=family.within_direction(q,p,n)
    # Differentiate the exact frozen-target loss through a nontrivial five-parameter map.
    W=torch.randn(3,4,19,5);params=torch.randn(5,requires_grad=True)
    live=W@params; direction=family.within_direction(live.detach(),p,n)
    loss=.5*(live-live.detach()-direction).square().sum(-1).mean()
    g=torch.autograd.grad(loss,params)[0]
    symmetries.update(scaled_swap=float((second+family.within_direction(q,n,p)).abs().max()),
                      binary_swap=float((r+model.within_formula_field(q,n,p)).abs().max()),
                      same_binary_scaled_field=error(r,second),
                      detached_gradient=error(g,-(W*direction[...,None]).sum(-2).mean((0,1))))
    assert max(symmetries.values())<1e-12
    return dict(witness=cases,symmetries_and_gradient=symmetries)


def distance_change():
    # Training-only data to model, and data to data, at the exact condition.
    torch.set_default_dtype(torch.float32)
    rng=np.random.default_rng(60321)
    with np.load(STUDY/"data/multi-train.npz") as src:
        z,k,f=[torch.from_numpy(src[name]) for name in ("z","k","f")]
    z=z.long();k=k.float();f=f.float()
    checkpoint=torch.load(STUDY/"drift-results/multi-exact/seed11.pt",weights_only=True)
    net=model.CrystalGenerator().eval();net.load_state_dict(checkpoint["model"])
    true=fingerprint((k-checkpoint["mean"])/checkpoint["scale"],f,z,"harmonic")
    _,labels=np.unique(z.numpy(),axis=0,return_inverse=True)
    groups={label:np.flatnonzero(labels==label) for label in set(labels)}
    ix=rng.integers(len(z),size=4096);other=np.array([rng.choice(groups[labels[i]]) for i in ix])
    torch.manual_seed(60321)
    with torch.no_grad():
        cell,pos=net(z[ix],torch.randn(len(ix),6),torch.rand(len(ix),8,3))
        gen=fingerprint(cell,pos,z[ix],"harmonic")
    summary={}
    for name,left,right in (("real_generated",true[ix],gen),("real_real",true[ix],true[other])):
        diff=left-right
        original=diff[:,:70].norm(dim=1);new=diff.norm(dim=1)
        active=original>1e-7
        percent=100*(new[active]/original[active]-1).numpy()
        summary[name]=dict(sampled=len(ix),nonzero=int(active.sum()),same_or_numerically_equal=int((~active).sum()),
                           median_pct=float(np.median(percent)),p95_pct=float(np.quantile(percent,.95)),
                           max_pct=float(percent.max()),mean_added_squared_distance=float(diff[:,-1].square().mean()),
                           redundant_channel_weight=float((diff[:,-1].square().mean()/diff[:,6].square().mean()).sqrt()))
    return summary


if __name__=="__main__":
    torch.set_num_threads(1)
    check=geometry_checks()
    check["training_feature_distance_change"]=distance_change()
    (HERE/"checks.json").write_text(json.dumps(check,indent=2)+"\n")
    print(json.dumps(check,indent=2))
