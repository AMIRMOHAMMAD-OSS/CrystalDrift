"""Numerically test the actual periodic features, finite-batch field and trained decoder API."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from lattice_symmetry.generator import PeriodicGenerator
from lattice_symmetry.kernel import periodic_features
from scaled.family import within_direction
from .physical import Generator,features,blocks
from .neighbors import edges,numpy_indices

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results"
DT=torch.float64
K=3
R=2
CUT=3.4
FAMILIES=((0,1,4,6),(0,1,3,7))


def maxerr(x,y):
    return float((x-y).abs().max().detach())


def stack(crystals,device="cpu",dtype=DT):
    n=max(len(z) for _,_,z in crystals)+2
    f=torch.zeros(len(crystals),n,3,dtype=dtype,device=device)
    z=torch.zeros(len(crystals),n,dtype=torch.long,device=device)
    for i,(_,pos,species) in enumerate(crystals):
        f[i,:len(pos)]=torch.as_tensor(pos,device=device,dtype=dtype)
        z[i,:len(pos)]=torch.as_tensor(species,device=device)
    l=torch.as_tensor(np.stack([x[0] for x in crystals]),device=device,dtype=dtype)
    return l,f,z


def records():
    rng=np.random.default_rng(230114)
    base=np.array([[3.91,.4,-.22],[.13,4.04,.1],[-.26,.18,4.32]])
    return [(base+.06*rng.normal(size=(3,3)),rng.uniform(0,1,(4,3)),np.array([8,14,6,8])) for _ in range(9)]


def change(record,rng,idx):
    l,f,z=record
    u=np.eye(3,dtype=float)
    for _ in range(2):
        a,b=rng.choice(3,2,replace=False)
        u[a]+=rng.choice([-3,-1,1,3])*u[b]
    if idx%2:u[[1,2]]=u[[2,1]]
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if (np.linalg.det(q)<0)!=(idx%3==0):q[:,0]*=-1
    p=rng.permutation(len(z)); shift=rng.uniform(-1,1,3)
    fp=f[p]@np.linalg.inv(u)+shift
    fp=fp-np.floor(fp)+rng.integers(-2,3,fp.shape)
    return (u@l@q,fp,z[p]),(u,q,p,shift)


def phi(batch):
    return features(*batch,cutoff=CUT,n_radial=R,species_slots=K)


def raw_affinity(q,p,n):
    d=torch.cdist(q,torch.cat((p,n),dim=1))
    return [(-d/t).exp() for t in (.3,.8)]


def objective(net,rows,device="cpu",dtype=DT):
    l,f,z=stack(rows,device,dtype)
    lout,fout=net(l,f,z)
    q=phi((lout[:3],fout[:3],z[:3]))[None,:,:]
    with torch.no_grad():
        p=phi((l[3:6],f[3:6],z[3:6]))[None,:,:]
        n=phi((lout[6:],fout[6:],z[6:]))[None,:,:]
        v=within_direction(q.detach(),p,n)
        swapped=within_direction(q.detach(),n,p)
    loss=.5*(q-(q.detach()+v)).square().sum(-1).mean()
    grad=torch.cat([g.flatten() for g in torch.autograd.grad(loss,net.parameters())])
    return dict(descriptor=torch.cat((q,p,n),1),affinities=torch.stack(raw_affinity(q.detach(),p,n)),
                field=v,gradient=grad,swapped=swapped,cell=lout,sites=fout,loss=loss)


def independent_numpy(record,cutoff=CUT,K=K,R=R):
    """Contract the addition-theorem polynomial directly over ordered edge pairs."""
    l,f,z=record
    species,rank=np.unique(z,return_inverse=True)
    centers=np.linspace(0,1,R)
    # This independently sized brute cube is used only for the small nonskew reference cells.
    radius=int(np.ceil(cutoff/np.linalg.svd(l,compute_uv=False).min()+np.abs(f[:,None]-f[None,:]).max()))+1
    shifts=np.array(list(itertools.product(range(-radius,radius+1),repeat=3)))
    shell=np.zeros(R); radial=np.zeros((K*(K+1)//2,R)); powers=np.zeros((3,K,K*R,K*R))
    for i in range(len(z)):
        rows=[]; weights=[]; labels=[]
        for j in range(len(z)):
            d=(f[j]-f[i]+shifts)@l
            inside=((d*d).sum(1)<cutoff**2)&((i!=j)|(shifts!=0).any(1))
            q=d[inside]/cutoff
            t=(q*q).sum(1)
            w=np.exp(1-1/(1-t))[:,None]*np.exp(-.5*((t[:,None]-centers)*max(R-1,1))**2)
            if i==j==0:shell+=w.sum(0)
            if i!=j:
                a,b=sorted((rank[i],rank[j]))
                radial[b*(b+1)//2+a]+=w.sum(0)/(2*len(z))
            for dx,wrow in zip(q,w):
                rows.append(dx);weights.append(wrow);labels.append(rank[j])
        if not rows:continue
        xyz=np.array(rows)
        dots=xyz@xyz.T
        sq=(xyz**2).sum(1)
        coeff=np.zeros((len(rows),K*R))
        for j,a in enumerate(labels):coeff[j,a*R:(a+1)*R]=weights[j]
        for ell,operation in enumerate((np.ones_like(dots),dots,(3*dots*dots-sq[:,None]*sq)/2)):
            powers[ell,rank[i]]+=coeff.T@operation@coeff/np.sum(rank==rank[i])
    ix,jx=np.triu_indices(K*R)
    return np.concatenate(([np.linalg.slogdet(l)[1]],np.bincount(z,minlength=119)[1:119],shell,radial.ravel(),
                           *(p[:,ix,jx].ravel() for p in powers)))


def witness():
    rec=[]
    for x in FAMILIES:
        az=2*np.pi*np.array(x)/12
        p=np.column_stack((np.cos(az),np.sin(az),np.zeros(4)))
        rec.append((np.eye(3)*16,.5+p/16,np.ones(4,dtype=int)*6))
    out={}
    def describe(rows,cut):
        batch=stack(rows)
        value=features(*batch,cutoff=cut,n_radial=2,species_slots=2)
        dif=(value[0]-value[1]).detach()
        return {name:float(dif[sl].norm()) for name,sl in blocks(2,2).items()}
    chords=[]
    for l,f,z in rec:
        edges6=[(np.linalg.norm((f[i]-f[j])@l),i,j) for i,j in itertools.combinations(range(4),2)]
        edges6.sort()
        chords.append(([float(x[0]) for x in edges6],bool(set(edges6[0][1:])&set(edges6[1][1:]))))
    out["twelve_gon"]=dict(cutoff=2.25,cell_edge=16,polygon_radius=1,vertices=FAMILIES,
                            norms=describe(rec,2.25),chord_lengths=chords)
    assert np.allclose(chords[0][0],chords[1][0]) and not chords[0][1] and chords[1][1]
    eight=[]
    for x in FAMILIES:
        f=np.zeros((8,3));f[:,0]=np.tile(np.array(x)/12,2);f[4:,1]=1/3
        eight.append((np.eye(3),f,np.array([8]*4+[14]*4)))
    out["eight_site"]=dict(cell_edge=1.,cutoff=4.,norms=describe(eight,4.))
    e=out["twelve_gon"]["norms"]
    assert e["radial"]<1e-12 and min(e["power1"],e["power2"])>1e-7
    assert out["eight_site"]["norms"]["radial"]<1e-10
    return out


def regress():
    torch.manual_seed(12)
    torch.set_num_threads(1)
    rows=records()
    rng=np.random.default_rng(51)
    altered,transforms=zip(*(change(row,rng,i) for i,row in enumerate(rows)))
    net=Generator(cutoff=CUT).double()
    a=objective(net,rows);b=objective(net,altered)
    error={k:maxerr(a[k],b[k]) for k in ("descriptor","affinities","field","gradient","loss")}
    assert max(error.values())<2e-8 and float(a["gradient"].norm())>1e-6
    error["bank_swap"]=maxerr(-a["field"],a["swapped"])
    assert error["bank_swap"]<1e-12 and float(a["field"].norm())>1e-2
    cerr=ferr=0.
    for i,(u,q,p,t) in enumerate(transforms):
        target=u@a["cell"][i].detach().numpy()@q
        cerr=max(cerr,float(np.max(np.abs(b["cell"][i].detach().numpy()-target))))
        delta=b["sites"][i,:4].detach().numpy()-a["sites"][i,p].detach().numpy()@np.linalg.inv(u)-t
        ferr=max(ferr,float(np.max(np.linalg.norm((delta-np.rint(delta))@target,axis=1))))
    assert max(cerr,ferr)<2e-9
    scalar=PeriodicGenerator(cutoff=CUT,width=24,n_radial=5).double()
    scalar.load_state_dict(net.state_dict(),strict=True)
    l,f,z=stack(rows)
    scal=scalar(l[0],f[0,:4],z[0,:4])
    ref=periodic_features(l[0],f[0,:4],z[0,:4],(6,8,14),CUT,R)
    val=phi((l,f,z))
    sr=blocks(R,K)["radial"]
    actual=torch.cat((val[0,:1],val[0,torch.tensor([6,8,14])],val[0,119:sr.stop]))
    scalar_error=max(maxerr(scal[0],a["cell"][0]),maxerr(scal[1],a["sites"][0,:4]),maxerr(actual,ref))
    assert scalar_error<2e-10
    oracle=np.stack([independent_numpy(r) for r in rows])
    oracle_error=float(np.max(np.abs(oracle-val.detach().numpy())))
    assert oracle_error<1e-10
    # Independent exhaustive cubes in the original basis; transform vectors rather than image labels.
    l1,f1,z1=stack(altered)
    ss,tt,bb,vv=edges(l1,f1,z1,CUT)
    N=z1.shape[1]
    max_neigh=0.
    for bidx,(base,(u,q,p,t)) in enumerate(zip(rows,transforms)):
        L,F,Z=base
        shifts=np.array(list(itertools.product(range(-3,4),repeat=3)))
        assert 3*np.linalg.svd(L,compute_uv=False).min()>CUT
        for i,j in itertools.product(range(4),repeat=2):
            vs=(F[p[j]]-F[p[i]]+shifts)@L@q
            valid=((vs*vs).sum(1)<CUT**2)&((i!=j)|(shifts!=0).any(1))
            expected=vs[valid]
            observed=vv[(bb==bidx)&(ss%N==i)&(tt%N==j)].numpy()
            assert len(expected)==len(observed)
            if len(expected):
                dist=np.linalg.norm(expected[:,None]-observed[None,:],axis=-1)
                max_neigh=max(max_neigh,float(max(dist.min(0).max(),dist.min(1).max())))
    assert max_neigh<1e-10
    # A first-site pad and internal vacancies may occur in an otherwise physical example.
    mixed=[rows[0],(2.4*np.eye(3),np.zeros((1,3)),np.array([29])),
           (12*np.eye(3),np.array([[0,0,0],[0,0,0],[1e-7,0,0]]),np.array([14,14,6]))]
    ml,mf,mz=stack(mixed)
    ml.requires_grad_(True);mf.requires_grad_(True)
    feat=phi((ml,mf,mz)); lc,fc=net(ml,mf,mz)
    gs=torch.autograd.grad(feat.sum()+lc.sum()+fc.sum(),(ml,mf))
    assert all(torch.isfinite(x).all() for x in (*gs,feat,lc,fc))
    mixed_oracle=float(np.max(np.abs(feat.detach().numpy()-np.stack([independent_numpy(r) for r in mixed]))))
    pp=torch.tensor([4,1,3,0,2,5])
    perr=maxerr(feat,phi((ml,mf[:,pp],mz[:,pp])))
    assert max(perr,mixed_oracle)<1e-10
    # Cache parity: a batch of detached positive rows must equal encoding each record alone.
    cached=torch.cat([phi(stack([r])).detach() for r in rows])
    err_cache=maxerr(val,cached)
    assert err_cache<1e-11
    def fd(x):
        j,p,z=stack([rows[0]])
        direction=torch.arange(j.numel(),dtype=DT).reshape_as(j)/20
        df=torch.linspace(.05,.3,p.numel(),dtype=DT).reshape_as(p)
        h=phi((j+x*direction,p+x*df,z))[0]
        return h@torch.sin(torch.arange(h.numel(),dtype=DT)+.7)
    h=torch.tensor(0.,dtype=DT,requires_grad=True)
    g=torch.autograd.grad(fd(h),h)[0].item()
    fdiff=float((fd(1e-5)-fd(-1e-5))/(2e-5))
    assert abs(g-fdiff)<1e-7 and abs(g)>1e-4
    a0,b0,c0=rows[0]
    changed=list(rows[0]);changed[0]=a0*1.04
    ang=list(rows[0]);ang[0]=a0.copy();ang[0][0,1]+=.33
    elem=list(rows[0]);elem[2]=c0.copy();elem[2][1]=7
    controls=[float((phi(stack([r]))-val[:1]).norm()) for r in (changed,ang,elem)]
    assert min(controls)>1e-3
    # Vacuum ambiguity and reflected tetrahedron of uniquely labeled centers.
    vac=[(np.eye(3)*24,np.array([[0.,0.,0.],[s,0.,0.]]),np.array([6,8])) for s in (.25,.375)]
    vv=phi(stack(vac))
    assert maxerr(vv[0],vv[1])<1e-14
    tetra=np.array([[1.,1.,1.],[2.,1.,1.],[1.1,2.,1.],[1.,1.2,2.3]])
    mirror=np.diag([-1.,1.,1.])
    ch=[(np.eye(3)*16,p/16,np.array([6,7,8,14])) for p in (tetra,tetra@mirror)]
    cv=features(*stack(ch),n_radial=2,species_slots=4)
    chirality=maxerr(cv[0],cv[1])
    vol=np.linalg.det(tetra[1:]-tetra[0])
    assert chirality<1e-12 and abs(vol)>1
    cutoff=[]
    for x in (CUT-1e-2,CUT,CUT+1e-2):
        c,p,z=stack([(np.eye(3)*20,np.array([[0.,0,0],[x/20,0,0]]),np.array([8,8]))])
        p.requires_grad_(True)
        y=phi((c,p,z))
        eval_=float(y[0,blocks(R,K)["radial"]].norm().detach())
        grad_=torch.autograd.grad(y.sum(),p)[0].abs().max().item()
        cutoff.append([x,eval_,grad_])
    assert max(x[1] for x in cutoff)<1e-40 and max(x[2] for x in cutoff)<1e-35
    raw=float(np.linalg.norm(altered[0][0]-rows[0][0]))
    return dict(independent_batch_errors=error,cell_equivariance=cerr,physical_site_error_angstrom=ferr,
                prototype_max_error=scalar_error,addition_theorem_oracle=oracle_error,brute_skew_edges_angstrom=max_neigh,
                cutoff_radial_values_derivatives=cutoff,finite_difference=dict(autograd=g,central=fdiff,error=abs(g-fdiff)),
                cache_error=err_cache,mixed_padding_error=perr,mixed_oracle_error=mixed_oracle,
                label_length_angle_controls=dict(length=controls[0],angle=controls[1],species=controls[2]),
                raw_equivalent_cell_frobenius=raw,raw_laplacian_tau08=math.exp(-raw/.8),finite_cutoff_collision=maxerr(vv[0],vv[1]),
                enantiomer_descriptor_collision=chirality,signed_tetrahedron_volume_six=vol,
                measured_gradient_norm=float(a["gradient"].norm()),measured_finite_batch_field_norm=float(a["field"].norm()),
                training_witness=witness())


def gpu():
    if not torch.cuda.is_available():raise RuntimeError("A CUDA PyTorch build and device are required.")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(12)
    net=Generator(cutoff=CUT)
    source=objective(net,records(),dtype=torch.float32)
    net=net.cuda()
    other=objective(net,records(),device="cuda",dtype=torch.float32)
    result={k:maxerr(source[k],other[k].cpu()) for k in ("descriptor","affinities","field","gradient","cell","sites")}
    assert max(result.values())<5e-4,result
    return dict(device=torch.cuda.get_device_name(),torch=torch.__version__,tf32=False,errors=result,
                max_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",action="store_true")
    args=p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    result=gpu() if args.gpu else regress()
    target=RESULTS/("physics-cuda.json" if args.gpu else "physics-acceptance.json")
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(target,result)


if __name__=="__main__":main()
