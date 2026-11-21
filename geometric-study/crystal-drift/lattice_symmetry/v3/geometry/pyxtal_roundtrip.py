"""Run public PyXtal from_seed/to_pymatgen on fresh, prospectively hash-ordered formula draws."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1];O=HERE/"audits"

def make():
    import torch
    from ..models.sampler import Sampler
    torch.set_num_threads(1)
    reg=json.loads((O/"arrangement-registration.json").read_text())
    exclude=set(reg["excluded_ids"])|{r["id"] for r in reg["registered"]}
    with np.load(ROOT/"scaled/mp20-data/val.npz") as d:
        zz=d["z"];lookup={str(mid):i for i,mid in enumerate(d["ids"])}
    ids=sorted(set(lookup)-exclude,key=lambda s:hashlib.sha256(("v3-pyxtal-roundtrip/"+s).encode()).digest())[:32]
    seeds=[];s=Sampler()
    for mid in ids:
        i=lookup[mid]
        seeds.extend(s.seeds(zz[i],draws=1,seed=np.random.SeedSequence([1723,i])))
    matrix=json.loads((V3/"evaluation/first-frozen.json").read_text())["methods"]
    out={};patterns=[json.dumps([[int(e),l] for e,l in sd.template]) for sd in seeds]
    for spec in matrix:
        net=None
        if "checkpoint" in spec:
            path=V3/spec["checkpoint"]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==spec["sha"]
            net=s.load(path)
        pred=s.apply(net,seeds)
        out["L_"+spec["tag"]]=np.stack([x[0] for x in pred])
        out["F_"+spec["tag"]]=np.stack([np.pad(x[1],((0,20-len(x[1])),(0,0))) for x in pred])
    np.savez_compressed(O/"pyxtal-fixtures.npz",ids=ids,groups=[q.group for q in seeds],patterns=patterns,
        Z=np.stack([np.pad(sd.Z,(0,20-len(sd.Z))) for sd in seeds]),tags=[x["tag"] for x in matrix],
        weights=[x.get("sha","zero-call") for x in matrix],**out)

def check():
    from pyxtal import pyxtal
    from pymatgen.core import Structure,Element
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import spglib
    from standard_match import SETTINGS
    from .subgroups import group_check,crosscheck_table,HALL
    crosscheck_table()
    assert version("pyxtal")=="1.0.9" and version("pymatgen")=="2024.4.13"
    matcher=StructureMatcher(**SETTINGS)
    alt=json.loads((HERE/"letter-tables.json").read_text())["groups"]
    def canonical(g,spec):
        m=alt[str(g)]
        return min(tuple(sorted((z,p[m[0].index(l)]) for z,l in spec)) for p in m)
    rows=[];summaries={}
    with np.load(O/"pyxtal-fixtures.npz") as data:
        ids,gg,tt,zz=data["ids"],data["groups"],data["patterns"],data["Z"]
        for tag in data["tags"]:
            LL,FF=data["L_"+tag],data["F_"+tag];begin=len(rows)
            for mid,g,t,Z,L,F in zip(ids,gg,tt,zz,LL,FF):
                g=int(g);t=json.loads(str(t));m=Z>0;Z=Z[m];F=F[m]
                source=Structure(L,Z,F)
                sg=group_check(L,F,Z,g,np.zeros(3),.01)
                r=dict(id=str(mid),tag=str(tag),input_group=g,input_hall=int(HALL[g-1]),
                       input_template=t,spglib=sg)
                try:
                    q=pyxtal();q.from_seed(source,tol=.01,style="pyxtal")
                    back=q.to_pymatgen()
                    pat=[(int(Element(site.specie).Z),site.wp.letter) for site in q.atom_sites]
                    outg=int(q.group.number)
                    score=matcher.get_rms_dist(source,back);rev=matcher.get_rms_dist(back,source)
                    r.update(pyxtal_group=outg,pyxtal_hall=int(q.group.hall_number),pyxtal_template=pat,
                        equivalent_template=outg==g and canonical(g,t)==canonical(g,pat),
                        rms=float(score[0]) if score else None,reverse_rms=float(rev[0]) if rev else None)
                    if score is None or rev is None:
                        # Diagnose public native-to-conventional failures independently of recognition.
                        def prim(s):
                            cell=spglib.standardize_cell((s.lattice.matrix,s.frac_coords,s.atomic_numbers),
                                to_primitive=True,no_idealize=False,symprec=.01)
                            return Structure(cell[0],cell[2],cell[1])
                        x,y=prim(source),prim(back)
                        retry=matcher.get_rms_dist(x,y);retry_rev=matcher.get_rms_dist(y,x)
                        r["reduction_diagnostic"]=dict(native_sites=len(source),returned_sites=len(back),
                            pmg_primitive_counts=[len(s.get_reduced_structure().get_primitive_structure()) for s in (source,back)],
                            spglib_primitive_counts=[len(x),len(y)],volume_per_atom=[s.volume/len(s) for s in (source,back)],
                            separately_spglib_reduced_rms=float(retry[0]) if retry else None,
                            separately_spglib_reduced_reverse_rms=float(retry_rev[0]) if retry_rev else None)
                except (ValueError,TypeError,RuntimeError,IndexError,AttributeError,AssertionError,KeyError) as exc:
                    r.update(error=type(exc).__name__+": "+str(exc),rms=None,reverse_rms=None)
                rows.append(r)
            segment=rows[begin:]
            summaries[str(tag)]=dict(N=len(segment),contains=sum(r["spglib"]["contains"] for r in segment),
                exact_ops=sum(r["spglib"]["exact_operations"] for r in segment),
                recovered_both_directions=sum(r["rms"] is not None and r["reverse_rms"] is not None for r in segment),
                pyxtal_errors=sum("error" in r for r in segment),
                same_detected_type=sum(r.get("pyxtal_group")==r["input_group"] for r in segment),
                same_hall=sum(r.get("pyxtal_hall")==r["input_hall"] for r in segment),
                equivalent_template=sum(r.get("equivalent_template",False) for r in segment))
    out=dict(selection="32 SHA-256(v3-pyxtal-roundtrip/material ID) validation rows, excluding archived panels, registered dev/confirm and all 512 arrangement-challenge registrants. One deterministic first formula-only draw per method, identical seed; no validation structures used in generation.",
             versions={k:version(k) for k in ("pyxtal","pymatgen","spglib")},summary=summaries,records=rows,
             regimes="native-basis spglib requested-group containment and equality, independent PyXtal maximal-group recognition and geometric round trip")
    (O/"pyxtal-roundtrip.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf-8")
    print(summaries,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("task",choices=["make","check"]);a=p.parse_args()
    (make if a.task=="make" else check)()
