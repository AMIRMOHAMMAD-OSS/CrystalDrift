"""Angular drifting with independently sampled positives within the supplied template."""
import hashlib
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import numpy as np
import torch
from charts import HERE,expand
from geometry import cohort,reference
from labels import ROOT,SETTINGS
from prior import Prior,occupation,records
from orbit_power import orbit_ids,variance
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v2.physical import Generator,features
from scaled.family import within_direction
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

def batch(items):
    n=max(len(x[2]) for x in items);b=len(items)
    l=np.stack([x[0] for x in items]);f=np.zeros((b,n,3));z=np.zeros((b,n),int)
    for i,(_,points,atoms) in enumerate(items):f[i,:len(atoms)]=points;z[i,:len(atoms)]=atoms
    return torch.tensor(l,dtype=torch.float32),torch.tensor(f,dtype=torch.float32),torch.tensor(z,dtype=torch.long)

def real(r):
    f,z=expand(r["group"],r["template"],r["u"])
    return np.array(r["cell"]),f,z

def draw(r,prior,rng):
    return prior.draw(r["group"],r["template"],rng)

def apply(model,items):
    l,f,z=batch(items)
    with torch.no_grad():l,f=model(l,f,z)
    return [(L.numpy().astype(float),F[Z>0].numpy().astype(float),Z[Z>0].numpy()) for L,F,Z in zip(l,f,z)]

def validation(model,prior):
    rng=np.random.default_rng(203702);matcher=StructureMatcher(**SETTINGS);res=[]
    for r in cohort("val")[:64]:
        score=[]
        if "error" not in r:
            target=reference(r,"val")
            proposals=apply(model,[draw(r,prior,rng) for _ in range(5)])
            for L,F,Z in proposals:
                y=matcher.get_rms_dist(target,Structure(L,Z,F))
                if y is not None:score.append(float(y[0]))
        res.append(min(score) if score else None)
    passed=[v for v in res if v is not None]
    return dict(hits=len(passed),n=len(res),rms=float(np.mean(passed)) if passed else 1.,per_target=res)

def signature(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def main(with_orbits=False,output=None,resume=False,stop=512):
    torch.set_num_threads(1)
    out=Path(output) if output else HERE/("training-orbit" if with_orbits else "training")
    out.mkdir(parents=True,exist_ok=True)
    rows=records();prior=Prior(rows)
    rng=np.random.default_rng(113718);torch.manual_seed(11)
    net=Generator();optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r["group"],occupation(r["template"])].append(i)
    sc=HERE.parent/"csp/scaling.npz"
    with np.load(sc) as f:mean=torch.from_numpy(f["mean"]);scale=torch.from_numpy(f["scale"])
    def layout(r):return orbit_ids(r["group"],r["template"])
    orbit_reference=[]
    if with_orbits:
        srng=np.random.default_rng(717)
        for indices in np.array_split(srng.choice(len(rows),512,replace=False),32):
            selected=[rows[i] for i in indices]
            with torch.no_grad():
                orbit_reference.append(variance(*batch([real(r) for r in selected]),[layout(r) for r in selected]))
        ref=torch.cat(orbit_reference)
        ormean=ref.mean(0);orscale=((ref-ormean)**2).sum(-1).mean().sqrt().clamp_min(.01)
        np.savez(out/"orbit-scaling.npz",mean=ormean.numpy(),scale=orscale.numpy())
    def phi(l,f,z,ids):
        base=(features(l,f,z)-mean)/scale
        if with_orbits:return torch.cat([base,(variance(l,f,z,ids)-ormean)/orscale],-1)
        return base
    cache={};history=[];start=perf_counter();first=0
    resume_file=out/"resume.pt"
    if resume:
        data=torch.load(resume_file,map_location="cpu",weights_only=True)
        assert data["orbit"]==with_orbits and data["scale_sha256"]==signature(sc)
        first=data["step"];history=data["history"];rng.bit_generator.state=data["rng"]
        net.load_state_dict(data["model"]);optimizer.load_state_dict(data["optimizer"])
    for step in range(first,513):
        if step in (0,64,256,512) and not (resume and step==first):
            qq=validation(net,prior)
            target=out/f"angular-template-{step}.pt"
            torch.save(dict(model=net.state_dict(),step=step,architecture="lattice_symmetry.v2.physical.Generator",
                            conditioning="Hall+species-labelled occupied Wyckoff letters",scale_hash=signature(sc)),target)
            history.append(dict(step=step,validation=qq,checkpoint=target.name,sha256=signature(target),
                                wall_s=perf_counter()-start))
            (out/"validation.json").write_text(json.dumps(history,indent=1)+"\n")
            print(step,qq["hits"],qq["rms"],flush=True)
            tmp=out/"resume-building.pt"
            torch.save(dict(step=step,orbit=with_orbits,scale_sha256=signature(sc),model=net.state_dict(),
                            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,history=history),tmp)
            tmp.replace(resume_file)
        if step==stop and stop<512:return
        if step==512:break
        positives=[];noise=[];indices=[]
        for ri in rng.choice(len(rows),4):
            r=rows[ri];members=groups[r["group"],occupation(r["template"])]
            pidx=rng.choice(members,3,replace=True)
            for idx in pidx:
                if idx not in cache:
                    with torch.no_grad():cache[idx]=phi(*batch([real(rows[idx])]),[layout(rows[idx])])[0]
                positives.append(cache[idx])
            noise.extend(draw(r,prior,rng) for _ in range(6))
            indices.extend([layout(r)]*6)
        # Within each template the three queries, three model references, and
        # three data references are mutually independent conditional on the label.
        l,f,z=batch(noise);outl,outf=net(l,f,z)
        val=phi(outl,outf,z,indices).reshape(4,6,-1)
        q=val[:,:3];n=val[:,3:].detach();p=torch.stack(positives).reshape_as(n)
        vec=within_direction(q.detach(),p,n)
        loss=(q-(q.detach()+vec)).square().sum(-1).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);optimizer.step()
    best=min(history,key=lambda h:(-h["validation"]["hits"],h["validation"]["rms"],h["step"]))
    result=dict(chosen=best["checkpoint"],chosen_sha256=best["sha256"],seed=11,
                training_targets=len(rows),template_groups=len(groups),singleton_groups=sum(len(a)==1 for a in groups.values()),
                updates=512,examples_per_update=12,group_sampling="uniform training rows with replacement; independent positives",
                parameter_count=sum(p.numel() for p in net.parameters()),scaling_sha256=signature(sc),wall_s=perf_counter()-start)
    (out/"selection.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--orbit",action="store_true");p.add_argument("--out",type=Path)
    p.add_argument("--resume",action="store_true");p.add_argument("--stop",type=int,choices=[0,64,256,512],default=512)
    a=p.parse_args();main(a.orbit,a.out,a.resume,a.stop)
