"""Retrospective group sensitivity on each target's best accepted five-draw geometry."""
import argparse
import json
import os
from pathlib import Path

for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[v]="1"
import numpy as np
import spglib
from data import unpack_gram
from ..evaluation.storage import pool

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits";V3=HERE.parent;ROOT=V3.parents[1]

def inspect(panel,tags):
    with np.load(OUT/f"reachable-{panel}.npz") as d:
        refgroup=dict(zip(d["ids"],d["truth_group"]));n=len(refgroup)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as d:
        sizes=dict(zip(d["ids"],np.sqrt(np.linalg.det(unpack_gram(d["k"])))))
    report={}
    for tag in tags:
        native=tag.startswith("uflow-")
        if native:
            arm=tag.removeprefix("uflow-");raw=V3/f"uflow/predictions/{arm}/frozen/{panel}"
            score=V3/f"uflow/scores/{arm}/frozen/{panel}"
        else:
            raw=pool(panel,tag);score=V3/f"evaluation/scores/{panel}/{tag}"
        with np.load(score/"outcomes.npz") as scored:
            score_ids=scored["ids"].astype(str);all_rms=scored["rms"][:,:5]
            assert score_ids.tolist()==list(refgroup)
        allrows=[];hits=[]
        for file in sorted(raw.glob("?????-?????.npz")):
            with np.load(file) as d:
                take=slice(len(allrows),len(allrows)+len(d["ids"]))
                assert np.array_equal(d["ids"],score_ids[take]);r=all_rms[take];L,F,Z=d["L"],d["F"],d["Z"]
                allrows.extend(d["ids"].tolist())
                for ix,mid in enumerate(d["ids"]):
                    if not np.isfinite(r[ix]).any():continue
                    j=int(np.nanargmin(r[ix]));z=Z[ix] if native else Z[ix,j];mask=z>0
                    mat=L[ix,j].astype(float);f=F[ix,j,mask].astype(float);target=int(refgroup[mid])
                    observed={}
                    factor=(sizes[mid]/abs(np.linalg.det(mat)))**(1/3)
                    for label,l in (("raw",mat),("target_volume_posthoc",factor*mat)):
                        for tolerance in (.01,.1):
                            dataset=spglib.get_symmetry_dataset((l,f,z[mask]),symprec=tolerance)
                            observed[f"{label}/{tolerance}"]=int(dataset.number) if dataset is not None else 0
                    hits.append(dict(id=str(mid),draw=j,reference=target,rms=float(r[ix,j]),volume_scale=factor,**observed))
        assert allrows==list(refgroup)
        summary={key:dict(exact=sum(row[key]>0 and row[key]==row["reference"] for row in hits),
                failed=sum(row[key]==0 for row in hits)) for key in
                ("raw/0.01","raw/0.1","target_volume_posthoc/0.01","target_volume_posthoc/0.1")}
        report[tag]=dict(targets=n,matches=len(hits),comparisons=summary,records=hits)
        print(panel,tag,len(hits),summary,flush=True)
    dest=OUT/f"matched-group-tolerances-{panel}.json"
    previous=json.loads(dest.read_text()) if dest.is_file() else {}
    previous.update(report);dest.write_text(json.dumps(previous,separators=(",",":"))+"\n",encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("panel",choices=("family","shared"));p.add_argument("tags",nargs="+")
    a=p.parse_args();inspect(a.panel,a.tags)
