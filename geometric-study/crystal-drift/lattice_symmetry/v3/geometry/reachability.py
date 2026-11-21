"""Retrospective formula-only template coverage. The generator never receives these labels."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from ..models.sampler import chart,Sampler,V3,V2,ROOT

HERE=Path(__file__).resolve().parent
OUT=HERE/"audits"
ALT=json.loads((HERE/"letter-tables.json").read_text())["groups"]


@lru_cache(None)
def attrs(g,l):
    m=chart(g,l)[0]
    return len(m),int(np.linalg.matrix_rank(m[0]))


@lru_cache(None)
def keys(g,pat):
    species=defaultdict(list)
    for z,l in pat:species[z].append(attrs(g,l))
    family=(g,tuple(sorted(tuple(sorted(x)) for x in species.values())))
    degree=sum(attrs(g,l)[1] for _,l in pat)
    letter_maps=ALT[str(g)]
    equivalent=min(tuple(sorted((z,alt[letter_maps[0].index(l)]) for z,l in pat)) for alt in letter_maps)
    return family,(g,equivalent),degree


def records(split):
    for file in sorted((V2/"wyckoff/labels").glob(split+"-*.gz")):
        with gzip.open(file,"rt") as data:
            for line in data:yield json.loads(line)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("panels",nargs="+",choices=["dev","confirm","family","shared","test","arrangement","arrangement-control"]);a=ap.parse_args()
    split=json.loads((V3/"protocol/splits.json").read_text());s=Sampler();OUT.mkdir(exist_ok=True)
    roles={"dev":"val","confirm":"val","family":"train","shared":"train","test":"test",
           "arrangement":"val","arrangement-control":"val"}
    for panel in a.panels:
        if panel.startswith("arrangement"):
            archive=json.loads((OUT/"arrangement-frozen.json").read_text())
            ids=archive["uncovered_ids" if panel=="arrangement" else "covered_ids"]
        else:ids=split[panel+"_ids"]
        n=len(ids);role=roles[panel]
        with np.load(ROOT/f"scaled/mp20-data/{role}.npz") as d:
            idx={str(v):i for i,v in enumerate(d["ids"])};z=d["z"]
        labels={r["id"]:r for r in records(role)};coh=set(ids)
        assert len(coh)==len(ids) and coh<=set(labels)
        fields=["canonical","equivalent","family","group","fixed"]
        flags={f:np.zeros((n,20),bool) for f in fields};known=np.zeros(n,bool)
        valid_candidates=np.zeros(n,np.int16);nonprimitive=np.zeros(n,bool);truth_group=np.zeros(n,np.int16)
        for i,mid in enumerate(ids):
            zz=np.sort(z[idx[mid]][z[idx[mid]]>0]);options=s.shortlist(tuple(zz),False)
            valid_candidates[i]=len(options);r=labels[mid];known[i]="error" not in r
            if known[i]:
                tg=r["group"];tp=tuple(sorted(tuple(x) for x in r["template"]));tf,te,td=keys(tg,tp)
                nonprimitive[i]=r["native_n"]!=r["primitive_n"];truth_group[i]=tg
            for j in range(20):
                g,p,donor=options[j%len(options)];f,e,degree=keys(g,p)
                flags["fixed"][i,j]=degree==0
                if known[i]:
                    flags["group"][i,j]=g==tg
                    flags["canonical"][i,j]=(g,p)==(tg,tp)
                    flags["equivalent"][i,j]=e==te;flags["family"][i,j]=f==tf
        assert np.all(~flags["canonical"]|flags["equivalent"])
        assert np.all(~flags["equivalent"]|flags["family"])
        checks={k:{f:int(np.any(x[:,:k],axis=1).sum()) for f,x in flags.items()} for k in [1,5,20]}
        np.savez_compressed(OUT/f"reachable-{panel}.npz",ids=ids,truth_available=known,
                            nonprimitive_input=nonprimitive,truth_group=truth_group,
                            distinct_shortlist=valid_candidates,**flags)
        result=dict(panel=panel,ids=n,truth_available=int(known.sum()),reach_by_k=checks,
                    nonprimitive_labelled=int(nonprimitive.sum()),
                    fixed_draws_by_k={k:int(flags["fixed"][:,:k].sum()) for k in [1,5,20]},
                    source="default protected train-only Sampler.shortlist; cyclic ordered draws",
                    equivalents="single simultaneous letter permutation tabulated in PyXtal 1.0.9; same group, labelled elements, all orbits retained")
        (OUT/f"reachable-{panel}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print("completed",panel,n)


if __name__=="__main__":
    main()
