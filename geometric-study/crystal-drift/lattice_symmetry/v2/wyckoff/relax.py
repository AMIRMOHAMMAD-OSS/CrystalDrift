"""Unconstrained CHGNet relaxation of the frozen first 16 template-condition targets per arm."""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
import numpy as np
from charts import HERE
from labels import ROOT
from geometry import cohort,quality
sys.path.insert(0,str(ROOT))
from data import pack_gram,unpack_gram
from expansion.physical import run as physical

def setup():
    physical.HERE=HERE/"relaxation"
    physical.initialize()

def run(arm,workers):
    study=HERE/"relaxation"
    for p in ("inputs","records","cifs"):(study/p).mkdir(parents=True,exist_ok=True)
    rows=cohort("test");jobs=[]
    with np.load(HERE/f"pools/{arm}.npz") as b:
        ls,fs,zs=b["L"][:16,0],b["F"][:16,0],b["Z"][:16]
    for i,(l,f,z) in enumerate(zip(ls,fs,zs)):
        if not np.isfinite(l).all():continue
        name=f"{arm}-{i:03}";mask=z>0
        rec=dict(id=name,role="candidate",arm=arm,target=rows[i]["id"],draw=1,
                 k=pack_gram(l@l.T).tolist(),f=f[mask].tolist(),z=z[mask].tolist())
        dest=study/"inputs"/(name+".json")
        if dest.exists():assert json.loads(dest.read_text())==rec
        else:dest.write_text(json.dumps(rec)+"\n")
        if not (study/"records"/(name+".json")).exists():jobs.append(name)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers,initializer=setup) as ex:
            for _ in ex.map(physical.run,jobs):pass
    summary=[]
    for i,r in enumerate(rows[:16]):
        name=f"{arm}-{i:03}"
        path=study/"records"/(name+".json")
        q=dict(id=r["id"],index=i,arm=arm)
        if path.exists():
            obj=json.loads(path.read_text());q.update(status=obj["status"],converged=obj["converged"],wall=obj["wall_seconds"])
            if "final_f" in obj:
                mask=zs[i]>0
                q.update(quality(np.linalg.cholesky(unpack_gram(obj["final_k"])),np.asarray(obj["final_f"]),
                             zs[i,mask],r,start=fs[i,mask]))
        else:q.update(status="no raw geometry",converged=False)
        summary.append(q)
    (study/(arm+".json")).write_text(json.dumps(summary,indent=1)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arms",nargs="+");p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    for arm in a.arms:run(arm,a.workers)
