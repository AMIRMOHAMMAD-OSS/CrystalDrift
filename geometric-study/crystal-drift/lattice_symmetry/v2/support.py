"""Bound the deformation reachable from seed draws on the training set alone."""
import hashlib
import json
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from .physical import Generator,features
from .neighbors import edges
from .csp.model import Prior, make_cell, ROOT
from .acceptance import HERE,RESULTS


def quantiles(x):
    return np.quantile(x,[0,.05,.5,.95,1]).tolist()


def audit():
    rng=np.random.default_rng(815233)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as pool:
        indices=rng.choice(len(pool["ids"]),2048,replace=False)
        ids,z,k,f=[pool[n][indices] for n in ("ids","z","k","f")]
    prior_path=HERE/"csp/prior.npz"
    trained=Prior.load(prior_path)
    kfit,ffit=trained.draw(z,rng)
    ref=make_cell(torch.from_numpy(k).double())
    official=ref.numpy()
    n=(z>0).sum(1)
    # Exact prototype law; orthogonal right factors can be omitted for the invariant statistics below.
    noise=rng.standard_normal((len(z),3,3))
    original=torch.matrix_exp(torch.from_numpy(.1*(noise+noise.transpose(0,2,1)))).numpy()*(18.*n[:,None,None])**(1/3)
    u=rng.random((*z.shape,3))*(z>0)[:,:,None]
    out=dict(ids=ids.tolist(),train_sha256=hashlib.sha256((ROOT/"scaled/mp20-data/train.npz").read_bytes()).hexdigest(),
             prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),seed=815233,
             quantiles=[0,.05,.5,.95,1],
             target_volume_per_atom=quantiles(np.linalg.det(official)/n),
             target_log_aspect=quantiles(np.log(np.linalg.cond(official))))
    for key,L,F in (("original",original,u),("train_fitted",make_cell(torch.from_numpy(kfit).double()).numpy(),ffit)):
        vdiff=np.abs(np.log(np.abs(np.linalg.det(L)/np.linalg.det(official))))
        adiff=np.abs(np.log(np.linalg.cond(L)/np.linalg.cond(official)))
        available=[];zero=0;wrong=0;applicable=0;pair_bounds=[]
        for start in range(0,len(z),64):
            Z=torch.from_numpy(z[start:start+64]).long()
            ls=torch.from_numpy(L[start:start+64]); fs=torch.from_numpy(F[start:start+64]).double()
            s,d,b,vec=edges(ls,fs,Z,4.)
            rho=torch.linalg.vector_norm(vec,dim=-1)
            mask=(1-(rho/4)**2).clamp_min(0)**3
            norms=ls.new_zeros(Z.numel()).index_add_(0,s,mask*rho)
            den=ls.new_ones(Z.numel()).index_add_(0,s,mask)
            bounds=(norms/den).reshape(Z.shape).numpy()
            active=Z.numpy()>0
            available.extend(bounds[active].tolist())
            for j in range(len(Z)):
                row=start+j
                if not bounds[j].any():zero+=1
                elems,counts=np.unique(z[row,active[j]],return_counts=True)
                ix=[np.where(z[row]==q)[0][0] for q in elems[counts==1]]
                if len(ix)<2:continue
                applicable+=1
                diff=F[row,ix]-f[row,ix]
                delta=diff[:,None,:]-diff[None,:,:]
                frac_lo=np.linalg.norm(delta-np.rint(delta),axis=-1)
                smin=np.linalg.svd(L[row],compute_uv=False).min()
                bounds_pair=bounds[j,ix][:,None]+bounds[j,ix][None,:]
                if np.any(smin*frac_lo>bounds_pair+1e-7):wrong+=1
                pair_bounds.extend(bounds_pair[np.triu_indices(len(ix),1)].tolist())
        out[key]=dict(seed_volume_per_atom=quantiles(np.abs(np.linalg.det(L))/n),seed_log_aspect=quantiles(np.log(np.linalg.cond(L))),
                      abs_logvol_mismatch=quantiles(vdiff),abs_logaspect_mismatch=quantiles(adiff),
                      exceeds_logvol_bound=int((vdiff>1).sum()),exceeds_logaspect_bound=int((adiff>.25).sum()),
                      uncentered_site_radius_bound_angstrom=quantiles(available),empty_seed_graphs=zero,
                      singleton_species_pair_eligible=applicable,site_pair_bound_excludes_target=wrong,
                      singleton_pair_radius_bound_angstrom=quantiles(pair_bounds))
    return out


def profile():
    torch.manual_seed(19)
    batch=32
    with np.load(ROOT/"scaled/mp20-data/train.npz") as data:
        ix=np.random.default_rng(21931).choice(len(data["ids"]),batch,replace=False)
        L=make_cell(torch.from_numpy(data["k"][ix]).float());F=torch.from_numpy(data["f"][ix]).float();Z=torch.from_numpy(data["z"][ix]).long()
    out=dict(batch=batch,threads=torch.get_num_threads(),indices=ix.tolist(),dtype=str(F.dtype))
    model=Generator()
    def bench(fn):
        fn()
        t=[]
        for _ in range(23):
            start=perf_counter(); fn(); t.append(1000*(perf_counter()-start))
        return dict(median_ms=float(np.median(t)),p90_ms=float(np.quantile(t,.9)))
    out["neighbor_index_and_vectors"]=bench(lambda:edges(L,F,Z,4.))
    out["one_forward"]=bench(lambda:model(L,F,Z))
    for label in ("radial","angular"):
        angular=label=="angular"
        out[label+"_features"]=bench(lambda:features(L,F,Z,angular=angular))
        def backward():
            m,p=model(L,F,Z); phi=features(m,p,Z,angular=angular)
            model.zero_grad(set_to_none=True);phi.square().sum().backward()
        out[label+"_generator_features_backward"]=bench(backward)
        records=[]
        def pack(t):
            records.append(t.untyped_storage().nbytes())
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):backward()
        out[label+"_saved_storage_bytes_summed_with_repeats"]=int(sum(records))
    return out


if __name__=="__main__":
    torch.set_num_threads(1);RESULTS.mkdir(exist_ok=True)
    result=audit();result["cpu_profile"]=profile()
    (RESULTS/"seed-support.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print({a:{k:r[k] for k in ("exceeds_logvol_bound","exceeds_logaspect_bound","site_pair_bound_excludes_target","singleton_species_pair_eligible")} for a,r in result.items() if a in ("original","train_fitted")})
