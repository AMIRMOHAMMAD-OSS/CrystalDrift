"""Audit native structural duplicates within protected training and across the public test partition."""
from collections import defaultdict
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1];sys.path.insert(0,str(ROOT))
from standard_match import make_structure
from lattice_symmetry.v3.protocol.lock import chemistry

def main():
    registry=json.loads((V3/"protocol/splits.json").read_text())
    train_set=set(registry["train_ids"])
    matcher=StructureMatcher(ltol=.01,stol=.05,angle_tol=.5,scale=False,primitive_cell=True,
                             attempt_supercell=False,allow_subset=False)
    data={}
    for split in ("train","test"):
        with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as src:
            data[split]={k:src[k] for k in src}
    @lru_cache(maxsize=1024)
    def structure(split,i):
        d=data[split];return make_structure(d["k"][i],d["f"][i],d["z"][i])
    reps=defaultdict(list);output=[];comparisons=0
    for split in ("train","test"):
        d=data[split]
        for i,(mid,z) in enumerate(zip(d["ids"],d["z"])):
            if split=="train" and mid not in train_set:continue
            key=chemistry(z);witness="";err=""
            for j in reps[key]:
                comparisons+=1
                try:yes=matcher.fit(structure(split,i),structure("train",j))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    err=type(exc).__name__;continue
                if yes:witness=str(data["train"]["ids"][j]);break
            output.append((split,mid,witness,err))
            if split=="train" and not witness:reps[key].append(i)
    dst=HERE/"duplicates.csv"
    with dst.open("w",newline="") as fp:
        w=csv.writer(fp);w.writerow(("split","id","earlier_training_match","matcher_error"));w.writerows(output)
    body={s:dict(rows=sum(r[0]==s for r in output),positive=sum(r[0]==s and bool(r[2]) for r in output),
                 errors=sum(r[0]==s and bool(r[3]) for r in output)) for s in ("train","test")}
    body.update(comparisons=comparisons,criteria="species-preserving primitive fit, scale=False, ltol=.01 stol=.05 angle=.5; first-representative catalog; ordered nontransitive matching")
    (HERE/"duplicates.json").write_text(json.dumps(body,indent=2));print(body,flush=True)
if __name__=="__main__":main()
