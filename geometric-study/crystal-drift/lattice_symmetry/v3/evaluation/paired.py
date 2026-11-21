"""Generate matched ordered pools across frozen updates from one common stream of crystal seeds."""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
V3=HERE.parent
ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.evaluation.predict import h,atomic_npz,array,setup_split,draw_seed,fit_selector
from lattice_symmetry.v3.models.sampler import Sampler, Seed, species_key
from lattice_symmetry.v3.models.network import pack
from lattice_symmetry.v3.evaluation.storage import pool

def seed_at(s,z,o,rng,donor=False):
    if not donor or not o[-1]:
        return s.draw(z,o,rng)
    ix=s.mid[o[-1]];src=s.target(ix)
    target_elements=species_key(z)[1]
    remap=dict(zip(s.rich[ix][2],target_elements))
    Z=np.array([int(remap[e]) for e in src.Z])
    pattern=tuple((int(remap[e]),l) for e,l in src.template)
    ratio=(s.design(z)-s.design(src.Z))@s.beta
    return Seed(src.L*np.exp(ratio/3),src.F.copy(),Z,src.orbit.copy(),src.J.copy(),src.group,pattern,o[-1],o[-1])

def seeds(s,z,k,rng,ranker=None,donor=False):
    z=z[z>0]
    choices=s.shortlist(tuple(np.sort(z)),False) if ranker is None else ranker(z)
    return [seed_at(s,z,choices[j%len(choices)],rng,donor) for j in range(k)]

def checkpoint_for(spec):
    p=spec.get("checkpoint")
    if not p:return None
    p=V3/p
    if spec.get("sha") and h(p)!=spec["sha"]:raise ValueError("checkpoint digest differs "+str(p))
    return p

def zerosave(path,ids,L,F,Z,src,tag,sig,clock):
    atomic_npz(path,ids=ids,L=L,F=F,Z=Z,group=src[0],donor=src[1],cell_donor=src[2],
               patterns=src[3],seconds=clock,signature=sig,tag=tag,errors=src[4])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("matrix",type=Path);ap.add_argument("--panel",required=True)
    ap.add_argument("--device",default="cpu");ap.add_argument("--chunk",type=int,default=96)
    ap.add_argument("--draws",type=int,default=20);ap.add_argument("--start",type=int,default=0);ap.add_argument("--stop",type=int)
    ap.add_argument("--inner",type=int,default=64);ap.add_argument("--threads",type=int,default=1)
    a=ap.parse_args(); torch.set_num_threads(a.threads);torch.set_float32_matmul_precision("highest")
    cohort,split=setup_split(a.panel)
    methods=json.loads(a.matrix.read_text())["methods"]
    if a.panel in ("test","family","shared") and not json.loads(a.matrix.read_text()).get("frozen"):
        raise ValueError("Final evaluation requires frozen register")
    s=Sampler(a.device)
    groups={}
    for spec in methods:
        tag=spec["tag"];p=checkpoint_for(spec);base=pool(a.panel,tag,write=True);base.mkdir(exist_ok=True,parents=True)
        sig=json.dumps(dict(weight=h(p) if p else "prior",family=spec.get("family","symbolic"),selector=spec.get("selector","lookup"),
                 spec_sha=h(V3/"protocol/splits.json"),draws=a.draws,seed_stream="draw_seed/20261121"),sort_keys=True)
        group=spec.get("family","symbolic"),spec.get("selector","lookup")
        groups.setdefault(group,[]).append((tag,s.load(p) if p else None,base,sig))
    for (family,selector),arms in groups.items():
        ranker=fit_selector(s,selector);rankfn=(lambda z:ranker.rank(z,selector)) if ranker is not None else None
        for i in range(a.start,min(a.stop or len(cohort),len(cohort)),a.chunk):
            rows=cohort[i:min(i+a.chunk,a.stop or len(cohort))];count=len(rows);kk=a.draws
            dest=f"{i:05}-{i+count:05}.npz"
            active=[]
            for tag,net,base,sig in arms:
                path=base/dest
                if path.exists():
                    with np.load(path) as old:
                        assert str(old["signature"])==sig and old["ids"].tolist()==[r[0] for r in rows]
                else:active.append((tag,net,base,sig))
            if not active:continue
            s.shortlist.cache_clear()
            nseed=[];ixout=[];pats=[];errs=[]
            gg=np.zeros((count,kk),np.int16);ds=np.full((count,kk),"",dtype="U22");cs=ds.copy()
            for j,(mid,ix,z) in enumerate(rows):
                rng=np.random.default_rng(draw_seed(a.panel,ix)); got=[]
                try:
                    got=seeds(s,z,kk,rng,rankfn,family=="donor")
                    assert all(np.array_equal(np.sort(q.Z),np.sort(z[z>0])) for q in got)
                    nseed.extend(got);ixout.extend(range(j*kk,(j+1)*kk))
                    gg[j]=[q.group for q in got];ds[j]=[q.donor for q in got];cs[j]=[q.cell_donor for q in got]
                    errs.append("")
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:errs.append(str(exc))
                pats.append(json.dumps([[(int(e),l) for e,l in q.template] for q in got],separators=(",",":")))
            payload={tag:array(count,kk) for tag,_,_,_ in active}
            clocks={tag:0. for tag,_,_,_ in active}
            with torch.inference_mode():
                for at in range(0,len(nseed),a.inner):
                    batch=nseed[at:at+a.inner]
                    tensors=pack(batch,a.device) if any(net is not None for _,net,_,_ in active) else None
                    for tag,net,_,_ in active:
                        tic=time.perf_counter()
                        if net is None:outs=[(q.L,q.F,q.Z) for q in batch]
                        else:
                            ll,ff=net(*tensors);ll=ll.cpu().numpy();ff=ff.cpu().numpy()
                            outs=[(l,f[:len(q.Z)],q.Z) for l,f,q in zip(ll,ff,batch)]
                        L,F,Z=payload[tag]
                        for idx,(l,f,z) in zip(ixout[at:at+len(batch)],outs):
                            row,col=divmod(idx,kk);L[row,col]=l;F[row,col,:len(z)]=f;Z[row,col,:len(z)]=z
                        clocks[tag]+=time.perf_counter()-tic
            for tag,net,base,sig in active:
                zerosave(base/dest,[r[0] for r in rows],*payload[tag],(gg,ds,cs,pats,errs),tag,sig,clocks[tag])
            if i%(a.chunk*10)==0:print(a.panel,family,selector,i+count,len(cohort),flush=True)
        for tag,_,base,sig in arms:
            (base/"settings.json").write_text(json.dumps(dict(split=split,panel=a.panel,count=len(cohort),
                draws=a.draws,signature=sig,device=a.device,batch=a.inner,threads=a.threads))+"\n")
if __name__=="__main__":main()
