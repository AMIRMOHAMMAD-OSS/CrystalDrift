"""Draw from one frozen formula-conditioned pipeline, saving independently replayable shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
V3=HERE.parent; ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.models.sampler import Sampler,weight_path
from data import pack_gram, unpack_gram

def h(path):
    return hashlib.sha256(weight_path(path).read_bytes()).hexdigest()

def atomic_npz(path,**kw):
    t=path.with_suffix(f".{os.getpid()}.part")
    with t.open("wb") as fp: np.savez_compressed(fp,**kw)
    t.replace(path)

def array(rows,draws):
    return (np.full((rows,draws,3,3),np.nan),np.full((rows,draws,20,3),np.nan,np.float32),
            np.zeros((rows,draws,20),np.int16))

def setup_split(panel):
    keys=json.loads((V3/"protocol/splits.json").read_text())
    split={"test":"test","dev":"val","confirm":"val","family":"train","shared":"train","clock":"test",
           "arrangement":"val","arrangement-control":"val"}[panel]
    with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as data: ids,z=data["ids"].astype(str),data["z"]
    if panel=="clock":
        selected=sorted(ids,key=lambda s:hashlib.sha256(("clock/"+s).encode()).digest())[:64]
    elif panel.startswith("arrangement"):
        record=json.loads((V3/"geometry/audits/arrangement-frozen.json").read_text())
        selected=record["covered_ids" if panel.endswith("control") else "uncovered_ids"]
    else:selected=keys[panel+"_ids"]
    lookup={mid:i for i,mid in enumerate(ids)}
    return [(s,lookup[s],z[lookup[s]]) for s in selected],split

def draw_seed(panel,ix):
    seed=38471 if panel in ("dev","confirm") else 94881 if panel in ("family","shared") else 771029
    return int(np.random.SeedSequence([seed,ix]).generate_state(1)[0])

def fit_selector(service,kind):
    if kind=="lookup":return None
    from lattice_symmetry.v3.evaluation.selector import Selector
    sel=Selector(service)
    if kind=="learned":
        sel.model.load_state_dict(torch.load(HERE/"selector/weights.pt",weights_only=True));sel.model.eval()
    return sel

def seeds_for(s,ranker,kind,z,k,seed):
    if ranker is None:return s.seeds(z,k,seed)
    z=z[z>0];opts=ranker.rank(z,kind);rng=np.random.default_rng(seed)
    return [s.draw(z,opts[i%len(opts)],rng) for i in range(k)]

def one(service,weights,ranker,args,rows,path,sig):
    count=len(rows);k=args.draws
    L,F,Z=array(count,k)
    groups=np.full((count,k),-1,np.int16)
    donors=np.full((count,k),"",dtype="U22");cell_donors=donors.copy()
    patterns=[];errors=[];ids=[];started=perf_counter()
    for i,(mid,ix,z) in enumerate(rows):
        ids.append(mid);sample_seed=draw_seed(args.panel,ix)
        try:
            seeds=seeds_for(service,ranker,args.selector,z,k,sample_seed)
            for j,sd in enumerate(seeds):
                groups[i,j]=sd.group;donors[i,j]=sd.donor;cell_donors[i,j]=sd.cell_donor
            patterns.append(json.dumps([[(int(e),l) for e,l in sd.template] for sd in seeds],separators=(",",":")))
            outs=service.apply(weights,seeds)
            for j,(cell,f,zout) in enumerate(outs):
                assert len(zout)==(z>0).sum() and np.array_equal(np.sort(zout),np.sort(z[z>0]))
                L[i,j]=cell; F[i,j,:len(zout)]=f; Z[i,j,:len(zout)]=zout
            errors.append("")
        except (FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
            if len(patterns)==i:patterns.append("[]")
            errors.append(str(exc))
    atomic_npz(path,ids=ids,L=L,F=F,Z=Z,group=groups,donor=donors,cell_donor=cell_donors,patterns=patterns,
               errors=errors,signature=sig,seconds=perf_counter()-started)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel",required=True,choices=["dev","confirm","family","shared","test","clock"])
    ap.add_argument("--tag",required=True)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--draws",type=int,default=20)
    ap.add_argument("--chunk",type=int,default=96)
    ap.add_argument("--selector",choices=["lookup","frequency","learned"],default="lookup")
    ap.add_argument("--device",default="cpu");ap.add_argument("--limit",type=int)
    a=ap.parse_args();torch.set_num_threads(1)
    cohort,split=setup_split(a.panel);cohort=cohort[:a.limit]
    out=HERE/"raw"/a.panel/a.tag;out.mkdir(parents=True,exist_ok=True)
    s=Sampler(a.device);w=s.load(a.checkpoint) if a.checkpoint else None;ranker=fit_selector(s,a.selector)
    sig=json.dumps(dict(weight=h(a.checkpoint) if a.checkpoint else "prior",split=h(V3/"protocol/splits.json"),
                        selector=a.selector,draws=a.draws),sort_keys=True)
    for i in range(0,len(cohort),a.chunk):
        rows=cohort[i:i+a.chunk];path=out/f"{i:05}-{i+len(rows):05}.npz"
        if path.is_file():
            with np.load(path) as p:
                if str(p["signature"])!=sig:raise ValueError("attempted to mix checkpoint or seed settings")
            continue
        one(s,w,ranker,a,rows,path,sig)
        if i%(a.chunk*10)==0:print(a.tag,i+len(rows),len(cohort),flush=True)
    (out/"complete.json").write_text(json.dumps(dict(split=split,count=len(cohort),draws=a.draws,signature=sig))+"\n")

if __name__=="__main__":main()
