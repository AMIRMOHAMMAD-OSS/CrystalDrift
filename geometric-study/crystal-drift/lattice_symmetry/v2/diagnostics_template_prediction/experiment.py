"""Frozen formula-only template-transfer pilot; no official test data are opened."""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[var]="1"

import numpy as np
import torch
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
import spglib

OUT=Path(__file__).resolve().parent
V2=OUT.parent
ROOT=V2.parents[1]
sys.path.insert(0,str(V2/"wyckoff"))
sys.path.insert(0,str(ROOT))
from charts import HALLS
from labels import SETTINGS
from prior import Prior, records
from evaluate import model_and_hash
from train_template import apply
from data import unpack_gram
from lattice_symmetry.v2.csp.model import load_service


def species_order(z):
    species, counts=np.unique(z[z>0],return_counts=True)
    chemistry=np.array([Element.from_Z(int(i)).mendeleev_no for i in species])
    order=np.lexsort((species,chemistry,counts))
    return tuple(counts[order]),species[order],chemistry[order]


def donor_index(rows, train_species):
    patterns=defaultdict(list)
    for r in rows:
        if r["native_n"] != r["primitive_n"]:
            continue
        sig, elements, chemistry=species_order(train_species[r["index"]])
        patterns[sig].append((r,elements,chemistry))
    return {sig:(rows,np.stack([v[1] for v in rows]),np.stack([v[2] for v in rows]))
            for sig,rows in patterns.items()}


def proposals(z, donors, k):
    sig, el, chem=species_order(z)
    if sig not in donors:
        return [(1,tuple((int(a),"a") for a in z if a),None)]*k
    group,elements,chemistry=donors[sig]
    # Stoichiometry matches exactly. Order equal-multiplicity roles by Mendeleev number.
    distances=(np.abs(chemistry-chem)+2*(elements!=el)).mean(axis=1)
    templates=[]
    seen=set()
    for i in np.argsort(distances,kind="stable"):
        row,src,unused=group[i]
        remap=dict(zip(src,el))
        pairs=tuple(sorted((int(remap[atom]),letter) for atom,letter in row["template"]))
        key=(int(row["group"]),pairs)
        if key not in seen:
            seen.add(key)
            templates.append((*key,row["id"]))
        if len(templates)==k:
            break
    return (templates*k)[:k]


def native(seed,g,z):
    # Convert with a centering basis determined by the predicted group alone.
    symbol=spglib.get_spacegroup_type(int(HALLS[g-1])).international_short[0]
    transforms={
        "P":np.eye(3),"I":np.array([[-1,1,1],[1,-1,1],[1,1,-1]])/2,
        "F":np.array([[0,1,1],[1,0,1],[1,1,0]])/2,
        "C":np.array([[1,1,0],[-1,1,0],[0,0,2]])/2,
        "A":np.array([[2,0,0],[0,1,1],[0,-1,1]])/2,
        "B":np.array([[1,0,1],[0,2,0],[-1,0,1]])/2,
        "R":np.array([[2,1,1],[-1,1,1],[-1,-2,1]])/3}
    L,F,Z=seed
    basis=transforms[symbol]
    f=(F@np.linalg.inv(basis))%1
    keep=[]
    for i in range(len(Z)):
        d=f[i]-f[keep]
        if not np.any((Z[keep]==Z[i]) & ((np.abs(d-np.rint(d))<1e-4).all(axis=1))):
            keep.append(i)
    if sorted(Z[keep])!=sorted(z[z>0]):
        raise ValueError("Expanded composition does not primitive-reduce to requested counts.")
    return Structure(basis@L,Z[keep],f[keep])


def main():
    torch.set_num_threads(1)
    rows=records()
    prior=Prior(rows)
    with np.load(ROOT/"scaled/mp20-data/train.npz",allow_pickle=False) as train:
        donors=donor_index(rows,train["z"])
        train_ids=set(train["ids"].tolist())
    c,meta=model_and_hash("angular-C")
    a=load_service("angular",11)
    matcher=StructureMatcher(**SETTINGS)
    cache=V2/"csp/fits/angular-11/val-00256.npz"
    with np.load(cache) as fitting:
        seen=set(fitting["ids"].astype(str))
    seen.update(x["id"] for x in json.loads((V2/"wyckoff/val-cohort.json").read_text()))
    with np.load(ROOT/"scaled/mp20-data/val.npz") as val:
        ids,zs=val["ids"],val["z"]
        rows_idx=sorted((i for i,s in enumerate(ids) if str(s) not in seen),
            key=lambda i:hashlib.sha256(("wyckoff-transfer-pilot-1:"+str(ids[i])).encode()).digest())[:192]
        assert not set(str(ids[i]) for i in rows_idx)&train_ids
        # Only IDs and input compositions are used in the sampler.
        geometry=[]
        for i in rows_idx:
            selected=proposals(zs[i],donors,5)
            rng=np.random.default_rng(np.random.SeedSequence([70693,int(i)]))
            seeds=[prior.draw(g,pairs,rng) for g,pairs,_ in selected]
            predicted=apply(c,seeds)
            k,f=a(np.repeat(zs[i:i+1],5,axis=0),
                  np.random.default_rng(np.random.SeedSequence([70693,int(i)])))
            angular=[Structure(np.linalg.cholesky(unpack_gram(x)),zs[i,zs[i]>0],y[zs[i]>0])
                     for x,y in zip(k,f)]
            geometry.append((selected,seeds,predicted,angular))
        # Validation geometry first accessed after all predictions are frozen.
        keys,positions=val["k"],val["f"]
        records_out=[]
        for i,(selected,seeds,pred,angular) in zip(rows_idx,geometry):
            active=zs[i]>0
            target=Structure(np.linalg.cholesky(unpack_gram(keys[i])),zs[i,active],positions[i,active])
            for j,(g,_,donor) in enumerate(selected):
                rec=dict(id=str(ids[i]),sites=int(active.sum()),draw=j+1,group=g,donor=donor)
                outs={"A":angular[j]}
                for label, geom in (("symbolic_prior",seeds[j]),("symbolic_angular",pred[j])):
                    try: outs[label]=native(geom,g,zs[i])
                    except ValueError as exc: rec[label+"_error"]=str(exc)
                for name in ("A","symbolic_prior","symbolic_angular"):
                    if name not in outs:
                        rec[name+"_rms"]=None
                        rec[name+"_logvolume"]=None
                        continue
                    proposal=outs[name]
                    match=matcher.get_rms_dist(proposal,target)
                    rec[name+"_rms"]=None if match is None else float(match[0])
                    reverse=matcher.get_rms_dist(target,proposal)
                    rec[name+"_rms_reverse"]=None if reverse is None else float(reverse[0])
                    rec[name+"_logvolume"]=float(np.log(proposal.volume/target.volume))
                records_out.append(rec)
    path=OUT/"draws.csv"
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in records_out))))
        writer.writeheader();writer.writerows(records_out)
    body={"target_count":len(rows_idx),"draws":5,"selection":"prefix of SHA256(wyckoff-transfer-pilot-1:material_id), excluding existing validation panels",
          "inputs":"element counts in native cell only","num_failed":{n:sum(n+"_error" in r for r in records_out) for n in ("symbolic_prior","symbolic_angular")},
          "matcher_order":"proposal first, target second, as in v2 formula-only reference. Reverse order retained in draws.csv for comparison with v2 supplied-template scoring.",
          "C_checkpoint":meta,"results":{}}
    for j in (1,5):
        hits={}
        for name in ("A","symbolic_prior","symbolic_angular"):
            res=[];vhit=[];best=[]
            for ix in range(0,len(records_out),5):
                group=records_out[ix:ix+j]
                matches=[r for r in group if r[name+"_rms"] is not None]
                res.append(bool(matches))
                vhit.append(any(abs(r[name+"_logvolume"])<=np.log(1.1) for r in matches))
                if matches:best.append(min(r[name+"_rms"] for r in matches))
            hits[name]=np.asarray(res,dtype=int)
            body["results"][f"{name}@{j}"]={"hits":sum(res),"volume_hits":sum(vhit),"matched_rms":np.mean(best)}
        rng=np.random.default_rng(111589)
        for left,right in (("symbolic_angular","symbolic_prior"),("symbolic_angular","A")):
            d=hits[left]-hits[right]
            ci=np.percentile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[2.5,97.5])
            body["results"][f"{left}-{right}@{j}"]={"delta":d.mean(),"ci95":ci.tolist()}
    (OUT/"summary.json").write_text(json.dumps(body,indent=2)+"\n")
    print(json.dumps(body,indent=2),flush=True)


if __name__=="__main__":
    main()
