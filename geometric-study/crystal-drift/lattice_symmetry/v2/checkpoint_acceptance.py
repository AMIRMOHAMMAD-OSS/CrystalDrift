"""Validate physical messages, unwrapped shifts, strain and training gradients after restore."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .acceptance import HERE,RESULTS,change,maxerr,objective,records,stack
from .csp.model import GENERATOR,DESCRIPTOR
from .physical import features,blocks
from scaled.family import within_direction
from .neighbors import edges
from .physical import Generator

def capture(net,rows):
    state={}
    def keep(name):
        def save(module,inputs,result):state[name]=result.detach()
        return save
    hooks=[net.edge.register_forward_hook(keep("edge")),net.dilation.register_forward_hook(keep("dilation"))]
    l,f,z=stack(rows)
    with torch.no_grad():net(l,f,z)
    for h in hooks:h.remove()
    B,N=z.shape
    s,d,b,r=edges(l,f,z,net.cutoff);q=r/net.cutoff
    raw=(1-q.square().sum(1,keepdim=True)).clamp_min(0)**3
    w=raw/(1+l.new_zeros(B*N,1).index_add(0,s,raw)[s])
    h=(w*state["edge"].tanh())
    n=(z>0).sum(1)
    u=l.new_zeros(B*N,3).index_add(0,s,h[:,:1]*r).reshape(B,N,3)
    v=(u-u.sum(1,keepdim=True)/n[:,None,None])*(z>0)[:,:,None]
    dyad=q[:,:,None]*q[:,None,:]*h[:,1:,None]
    e=.25*(state["dilation"].tanh()[:,:,None]*torch.eye(3)+l.new_zeros(B,3,3).index_add(0,b,dyad)/n[:,None,None])
    return v.numpy(),e.numpy()


def actual_bank_gradient(net,arm,rows):
    l,f,z=stack(rows)
    angular=arm!="radial"
    slices=blocks(angular=angular)
    dimension=slices["power0"].stop if arm=="power0" else max(x.stop for x in slices.values())
    with np.load(HERE/"csp/scaling.npz") as scale:
        mean,sd=[torch.from_numpy(scale[name][:dimension].copy()).double() for name in ("mean","scale")]
    def encode(L,F,Z):
        return ((features(L,F,Z,angular=angular,**DESCRIPTOR)[:,:dimension]-mean)/sd).reshape(8,4,dimension)
    with torch.no_grad():
        p=encode(l[32:64],f[32:64],z[32:64])
        ln,fn=net(l[64:],f[64:],z[64:])
        n=encode(ln,fn,z[64:])
    lq,fq=net(l[:32],f[:32],z[:32])
    q=encode(lq,fq,z[:32])
    d=within_direction(q.detach(),p,n)
    loss=.5*(q-q.detach()-d).square().sum(-1).mean()
    grad=torch.cat([g.flatten() for g in torch.autograd.grad(loss,net.parameters())])
    return d,grad


def verify(arm,seed,updates):
    checkpoint=HERE/"csp/fits"/f"{arm}-{seed}"/f"update-{updates:05}.pt"
    if arm not in ("angular","radial","power0"):raise ValueError("Physical generator required.")
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert json.loads(checkpoint.with_suffix(".json").read_text())["checkpoint_sha256"]==digest
    c=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert c["arm"]==arm and c["seed"]==seed and c["updates"]==updates
    net=Generator(**GENERATOR).double()
    net.load_state_dict(c["model"],strict=True)
    rows=records(); rng=np.random.default_rng(28162)
    other,trans=zip(*(change(row,rng,i) for i,row in enumerate(rows)))
    r=objective(net,rows);rr=objective(net,other)
    training_rows=[rows[i%len(rows)] for i in range(96)]
    altered=[change(row,rng,i)[0] for i,row in enumerate(training_rows)]
    old_d,old_grad=actual_bank_gradient(net,arm,training_rows)
    new_d,new_grad=actual_bank_gradient(net,arm,altered)
    drift=maxerr(old_d,new_d);grad=maxerr(old_grad,new_grad)
    assert old_d.norm().item()>0 and old_grad.norm().item()>0
    v,e=capture(net,rows);vt,et=capture(net,other)
    derr=terr=cerr=ferr=0.
    for i,(u,q,p,t) in enumerate(trans):
        derr=max(derr,float(np.abs(v[i,p]@q-vt[i,:4]).max()))
        terr=max(terr,float(np.abs(q.T@e[i]@q-et[i]).max()))
        a=u@r["cell"][i].detach().numpy()@q
        cerr=max(cerr,float(np.abs(a-rr["cell"][i].detach().numpy()).max()))
        df=rr["sites"][i,:4].detach().numpy()-r["sites"][i,p].detach().numpy()@np.linalg.inv(u)-t
        ferr=max(ferr,float(np.max(np.linalg.norm((df-np.rint(df))@a,axis=1))))
    assert max(drift,grad,derr,terr,cerr,ferr)<1e-7
    return dict(checkpoint_sha256=digest,arm=arm,seed=seed,updates=updates,field=drift,
                parameter_gradient=grad,cartesian_shift=derr,strain=terr,cell=cerr,sites_angstrom=ferr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",nargs="+",default=["angular","radial","power0"])
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    p.add_argument("--updates",type=int,default=256)
    p.add_argument("--selected",action="store_true")
    a=p.parse_args();torch.set_num_threads(1)
    lock=HERE/"csp/selection.json"
    if a.selected:
        sel=json.loads(lock.read_text())
        selected={arm:sel[arm]["updates"] for arm in a.arm}
        tag="selected"
    else:
        selected={arm:a.updates for arm in a.arm}
        tag=f"update-{a.updates}"
    result=[verify(arm,seed,selected[arm]) for arm in a.arm for seed in a.seeds]
    report=dict(architecture="lattice_symmetry.v2.physical.Generator",results=result,
                maximum_error=float(max(v for r in result for k,v in r.items() if k in ("field","parameter_gradient","cartesian_shift","strain","cell","sites_angstrom"))))
    if a.selected:report["selection_sha256"]=hashlib.sha256(lock.read_bytes()).hexdigest()
    (RESULTS/f"checkpoint-symmetry-{tag}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["maximum_error"],len(result),"restored checkpoints")


if __name__=="__main__":main()
