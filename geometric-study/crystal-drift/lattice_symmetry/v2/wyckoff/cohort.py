"""Freeze target IDs, independently check labels and audit detection tolerance."""
import argparse
import hashlib
import json
import os
for var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[var]="1"

import numpy as np
import spglib
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from charts import HERE,expand,label
from labels import dataset,SETTINGS

def select(split,n=128):
    ids,cells,fs,zs=dataset(split)
    order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(str(ids[i]).encode()).digest())[:n]
    out=[]
    for i in order:
        z=zs[i,zs[i]>0];f=fs[i,zs[i]>0];cell=cells[i]
        r=dict(id=str(ids[i]),index=i)
        try:
            r.update(label(cell,f,z))
            rf,rz=expand(r["group"],r["template"],r["u"])
            rms=StructureMatcher(**SETTINGS).get_rms_dist(Structure(cell,z,f),Structure(r["cell"],rz,rf))
            if rms is None:raise ValueError("StructureMatcher-roundtrip")
            r["roundtrip_rms"]=rms[0]
        except Exception as e:r["error"]=str(e)
        r["symprec"]={}
        for tol in [.001,.01,.1]:
            d=spglib.get_symmetry_dataset((cell,f,z),symprec=tol)
            r["symprec"][str(tol)]=None if d is None else int(d.number)
        out.append(r)
    (HERE/f"{split}-cohort.json").write_text(json.dumps(out,separators=(",",":"))+"\n")
    print(split,len(out),sum("error" not in r for r in out),flush=True)

if __name__=="__main__":
    select("val");select("test")
