"""Exercise the trained orbit-token map after changing all crystal descriptions independently."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from .descriptors import combine, drift_loss, feature_blocks, fit_stats
from .verify import HERE, change, diff
from ..models.network import OrbitNet, pack
from ..models.sampler import GROUP, M, T, START, compose, group_ops

PATTERNS = {
    1: ((3,"a"),(3,"a"),(8,"a")), 2: ((3,"i"),(3,"i"),(8,"a")),
    4: ((3,"a"),(3,"a"),(8,"a")), 14: ((3,"e"),(8,"a")),
    19: ((3,"a"),(8,"a")), 62: ((3,"c"),(8,"a"),(8,"c")),
    166: ((3,"c"),(8,"a"),(3,"c")), 225: ((3,"e"),(8,"a")),
    230: ((3,"a"),(8,"b"))}


def fixtures():
    rng = np.random.default_rng(419911)
    seeds = []
    for g,pat in PATTERNS.items():
        rot,_,_ = group_ops(g)
        L = np.array([[5.8,.5,-.6],[.7,6.5,.2],[.15,-.3,6.9]])
        H = (rot@(L@L.T)@rot.transpose(0,2,1)).mean(0)
        seeds.append(compose(g,pat,np.linalg.cholesky(H),rng.random((len(pat),3))))
    return seeds


def transform(seed,i,rng):
    (L,F,Z,_),(U,Q,p,t) = change((seed.L,seed.F,seed.Z,seed.orbit),rng,i)
    # pack assigns dense active tokens; give repeated charts a different token ordinal.
    names = rng.permutation(len(np.unique(seed.orbit)))
    J = seed.J[p] @ np.linalg.inv(U)
    # An invertible free-parameter rechart changes J on the left, possibly rank deficient.
    gauge = np.eye(3);gauge[0,1]=.5;gauge[2,0]=-.5
    return replace(seed,L=L,F=F,Z=Z,orbit=names[seed.orbit[p]],J=gauge@J),(U,Q,p,t)


def residual(seed, L, F):
    g=seed.group
    k=GROUP[g]-1;sl=slice(START[k],START[k+1])
    _,A,inv = group_ops(g)
    R=A@M[sl]@inv; translations=T[sl]@inv
    # Witness group operations in the native primitive cell. Use per-orbit sets (stronger
    # than bare species equality when two distinct occupied orbits have the same element).
    physical=np.linalg.inv(L) @ R @ L
    iso=np.max(np.abs(physical@physical.transpose(0,2,1)-np.eye(3)))
    pos=0.
    for o in np.unique(seed.orbit):
        motif=F[seed.orbit==o]
        actual=motif[None]@R + translations[:,None,:]
        dx=actual[:,:,None]-motif[None,None,:,:]
        distances=np.linalg.norm((dx-np.rint(dx))@L,axis=-1)
        pos=max(pos,float(distances.min(axis=2).max()),float(distances.min(axis=1).max()))
    return iso,pos


def evaluate(model,dtype,device):
    orig=fixtures();rng=np.random.default_rng(600411)
    alt,actions=zip(*(transform(s,i,rng) for i,s in enumerate(orig)))
    a=pack(orig,device,dtype); b=pack(alt,device,dtype)
    L,F=model(*a);Lp,Fp=model(*b)
    dl,df=model(*a,residual=True)
    maxima=dict(cell=0.,torus=0.,chart_tangent=0.,fixed_motion=0.,subgroup_metric=0.,
                subgroup_motif_angstrom=0.,volume_factor_excess=0.,symmetric_shape=0.)
    minimum_shape_stretch=1.; minimum_gram_eig=float("inf");n_fixed=0
    for i,(s,(u,q,p,t)) in enumerate(zip(orig,actions)):
        n=len(s.Z)
        U=L.new_tensor(u); Q=L.new_tensor(q)
        origin=F.new_tensor(t)
        expect=F[i,:n][p]@torch.linalg.inv(U)+origin
        maxima["cell"]=max(maxima["cell"],diff(U@L[i]@Q,Lp[i]))
        maxima["torus"]=max(maxima["torus"],float(((Fp[i,:n]-expect+.5).remainder(1)-.5).abs().max()))
        lattice=L[i].detach().cpu().numpy().astype(float)
        sites=F[i,:n].detach().cpu().numpy().astype(float)
        metric,motif=residual(s,lattice,sites)
        maxima["subgroup_metric"]=max(maxima["subgroup_metric"],metric)
        maxima["subgroup_motif_angstrom"]=max(maxima["subgroup_motif_angstrom"],motif)
        native=np.linalg.solve(s.L,lattice);scale=np.cbrt(abs(np.linalg.det(native)))
        shape=native/scale
        minimum_shape_stretch=min(minimum_shape_stretch,np.linalg.eigvalsh(shape).min())
        minimum_gram_eig=min(minimum_gram_eig,np.linalg.eigvalsh(lattice@lattice.T).min())
        maxima["symmetric_shape"]=max(maxima["symmetric_shape"],float(np.abs(shape-shape.T).max()))
        maxima["volume_factor_excess"]=max(maxima["volume_factor_excess"],abs(3*np.log(scale))-.6)
        shift=df[i,:n].detach().cpu().numpy().astype(float)
        for label in np.unique(s.orbit):
            mask=s.orbit==label;J=s.J[mask].transpose(1,0,2).reshape(3,-1)
            delta=shift[mask].flatten()
            fit=np.linalg.lstsq(J.T,delta,rcond=1e-12)[0]@J
            maxima["chart_tangent"]=max(maxima["chart_tangent"],float(np.abs(fit-delta).max()))
            if not np.any(J):
                n_fixed+=1
                maxima["fixed_motion"]=max(maxima["fixed_motion"],float(np.abs(delta).max()))
    gresults={}
    # Each bank row is an independent representation of its original. For this pure
    # implementation test use a shared chemistry and template across each synthetic bank.
    uniform=[orig[2]]*9
    for i,s in enumerate(uniform):
        uniform[i]=compose(s.group,s.template,s.L*(1+.015*i),
                           np.random.default_rng(40400+i).random((len(s.template),3)))
    other=[transform(s,i,rng)[0] for i,s in enumerate(uniform)]
    fixed=pack(uniform,device,dtype)
    changed=pack(other,device,dtype)
    for arm in ("contacts","reciprocal"):
        stats=fit_stats(feature_blocks(*fixed[:4],arm=arm))
        def grad(args):
            l,f=model(*args)
            x=combine(feature_blocks(l,f,*args[2:4],arm=arm),stats)
            q,p,n=x.reshape(3,3,-1).unbind(0)
            loss=drift_loss(q[None],p[None],n[None])
            return torch.cat([v.flatten() for v in torch.autograd.grad(loss,model.parameters())])
        g0,g1=grad(fixed),grad(changed)
        gresults[arm]=float((g0-g1).norm()/g0.norm().clamp_min(1e-10))
    tol = 3e-9 if dtype==torch.float64 else 3e-3
    assert max(maxima.values()) < tol,(dtype,maxima)
    assert max(gresults.values()) < tol,(dtype,gresults)
    assert minimum_shape_stretch>0 and minimum_gram_eig>0 and n_fixed>=2
    return dict(maxima=maxima,relative_parameter_gradient=gresults,minimum_shape_eigenvalue=minimum_shape_stretch,
                minimum_gram_eigenvalue=minimum_gram_eig,fixed_orbits=n_fixed,absolute_tolerance=tol)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path);p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    torch.manual_seed(8771);torch.set_num_threads(1)
    if a.device=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.checkpoint:
        record=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
        model=OrbitNet(**record["architecture"]);model.load_state_dict(record["net"])
        key=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    else:
        model=OrbitNet(width=120,depth=3);key="random-init-seed-8771"
    results=dict(device=a.device,torch=torch.__version__,numpy=np.__version__,
                 checkpoint_sha256_or_init=key,parameters=sum(p.numel() for p in model.parameters()),
                 tested_groups=list(PATTERNS),by_precision={})
    for dt in (torch.float64,torch.float32):
        results["by_precision"][str(dt)]=evaluate(model.to(device=a.device,dtype=dt),dt,a.device)
    a.output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(a.output)


if __name__=="__main__":
    main()
