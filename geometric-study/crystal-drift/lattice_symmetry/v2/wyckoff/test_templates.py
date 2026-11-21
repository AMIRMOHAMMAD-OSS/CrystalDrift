"""Independent crystallographic and paired-setting regressions for affine Wyckoff seeds."""
import json
import os
import sys
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
from pathlib import Path
import numpy as np
import torch
from charts import HERE,GROUPS,LETTERS,chart,operations,compatible_metric,expand,generic_draw,xyz
from orbit_power import variance,orbit_ids
from geometry import quality
from train_template import Generator,batch

def distance(a,b,L):
    delta=(a[:,None]-b[None,:]+.5)%1-.5
    dist=np.linalg.norm(delta@L,axis=-1)
    return max(dist.min(0).max(),dist.min(1).max())

def residual(L,F,Z,ops):
    rot,t=ops
    G=L@L.T
    err=np.abs(rot@G@rot.transpose(0,2,1)-G).max()/abs(G).max()
    max_site=0.
    for a in np.unique(Z):
        loc=F[Z==a]
        for r,tr in zip(rot,t):
            max_site=max(max_site,distance(loc@r+tr,loc,L))
    return float(err),float(max_site)

def audit():
    torch.set_num_threads(1);torch.manual_seed(441);rng=np.random.default_rng(71218)
    net=Generator().double()
    chosen=json.loads((HERE/"training/selection.json").read_text())
    state=torch.load(HERE/"training"/chosen["chosen"],map_location="cpu",weights_only=True)
    net.load_state_dict(state["model"])
    # Fixed inversion center, fourfold line, mirror plane, screw, glide, C centering,
    # and repeated compatible orbits of the same chemical species.
    cases=[(2,[(8,"a"),(11,"i")]),(75,[(12,"a"),(8,"d")]),(6,[(18,"a"),(8,"c")]),
           (4,[(11,"a"),(14,"a")]),(7,[(9,"a"),(15,"a")]),(5,[(8,"a"),(18,"c")]),
           (1,[(13,"a"),(13,"a"),(13,"a"),(13,"a")])]
    while len(cases)<39:
        g=int(rng.integers(1,231));letter=LETTERS[int(rng.integers(GROUPS[g]-GROUPS[g-1]))]
        m,t,free=chart(g,letter)
        if len(m)<=48:cases.append((g,[(14,letter)]))
    rows=[];optimizer=torch.optim.SGD(net.parameters(),lr=1e-3)
    for i,(g,spec) in enumerate(cases):
        u=generic_draw(g,spec,rng);f,z=expand(g,spec,u);orbit=orbit_ids(g,spec)
        a=rng.normal(size=(3,3));G=compatible_metric(g,np.eye(3)+a@a.T)
        L=np.linalg.cholesky(G);L*=np.cbrt(23*len(z)/np.linalg.det(L))
        op=operations(g);base=residual(L,f,z,op)
        U=np.array([[1,2,0],[0,1,1],[0,0,-1]]);Ui=np.linalg.inv(U)
        t0=rng.uniform(size=3);Q,_=np.linalg.qr(rng.normal(size=(3,3)));perm=rng.permutation(len(f))
        ll=U@L@Q;ff=(f[perm]@Ui+t0)%1.;zz=z[perm]
        rr=U@op[0]@Ui;tt=op[1]@Ui+t0-t0@rr
        rebase=residual(ll,ff,zz,(rr,tt))
        def tens(L,F,Z):return (torch.from_numpy(L)[None],torch.from_numpy(F)[None],torch.as_tensor(Z,dtype=torch.long)[None])
        A=tens(L,f,z);B=tens(ll,ff,zz);al,af=net(*A);bl,bf=net(*B)
        after=residual(al[0].detach().numpy(),af[0].detach().numpy(),z,op)
        d=(bf[0].detach().numpy()-(af[0].detach().numpy()[perm]@Ui+t0)+.5)%1-.5
        paired=max(np.abs(bl[0].detach().numpy()-U@al[0].detach().numpy()@Q).max(),abs(d).max())
        c1=variance(*A,[orbit]);c2=variance(*B,[rng.permutation(len(spec))[orbit[perm]]])
        orerr=float((c1-c2).detach().abs().max())
        # A real gradient step at fixed group leaves both metric and motif constraints intact.
        if i==0:
            loss=al.square().mean()+torch.cos(2*torch.pi*af).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        optL,optF=net(*A)
        opt=residual(optL[0].detach().numpy(),optF[0].detach().numpy(),z,op)
        detected=quality(L,f,z,dict(group=g,template=spec))
        assert max(*base,*rebase,*after,paired,orerr,*opt)<2e-7,(i,base,rebase,after,paired,orerr,opt)
        assert detected["occupancy0.001"] and detected["direct0.001"],(i,detected)
        rows.append(dict(group=g,template=spec,dof=[int(chart(g,l)[2].sum()) for _,l in spec],
            n=len(f),metric=base[0],sites=base[1],setting_site=rebase[1],after=after[1],
            optimized=opt[1],paired=float(paired),orbit_error=orerr,orbit_values=c1.detach().numpy().ravel().tolist(),
            detected=detected["number0.01"],max_position_change=float(abs((af.detach().numpy()[0]-f+.5)%1-.5).max())))
    assert set(d for r in rows for d in r["dof"])=={0,1,2,3}
    try:
        generic_draw(2,[(8,"a"),(14,"a")],rng)
    except ValueError as exc:
        assert "collision or specialization" in str(exc)
    else:raise AssertionError("A doubled fixed inversion-center occupation must be rejected")
    special=xyz(2,"i",np.zeros(3))
    assert len(special)==2 and len(np.unique(special,axis=0))==1
    # Test the full tabulation independently for generic multiplicity and coset
    # closure via a representative under the *spglib* Hall operation database.
    worst=0.;positions=0
    for g in range(1,231):
        rot,t=operations(g)
        for j in range(GROUPS[g]-GROUPS[g-1]):
            letter=LETTERS[j];u=rng.random(3)
            F=xyz(g,letter,u);images=(F[0]@rot+t)%1.
            e=distance(F,images,np.eye(3))
            assert e<1e-8 and len(np.unique(np.round(F%1,9),axis=0))==len(F),(g,letter,e)
            worst=max(worst,e);positions+=1
    out=dict(checkpoint=chosen["chosen_sha256"],positions=positions,hall_groups=230,worst_coset_error=float(worst),
             rejected_doubled_fixed_site=True,explicit_general_to_special_multiplicity=[2,1],
             maximum_paired=max(r["paired"] for r in rows),maximum_orbit_relabelling=max(r["orbit_error"] for r in rows),
             templates=rows)
    (HERE/"template-tests.json").write_text(json.dumps(out,indent=1)+"\n")
    print(positions,len(rows),out["maximum_paired"])

if __name__=="__main__":audit()
