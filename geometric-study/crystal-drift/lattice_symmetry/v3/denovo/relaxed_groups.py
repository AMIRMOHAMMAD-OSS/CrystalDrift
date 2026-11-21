"""Detect actual groups before/after CHGNet on the locked panel, including unrestricted proposals."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
import spglib
from data import unpack_gram

D=Path(__file__).resolve().parent

def detect(k,f,z,tol):
    L=np.linalg.cholesky(unpack_gram(k))
    data=spglib.get_symmetry_dataset((L,np.asarray(f),np.asarray(z)),symprec=tol)
    return int(data.number) if data is not None else 0

def audit(arm):
    registered=json.loads((D/"screen/panel.json").read_text())["indices_zero_based"]
    rows=[]
    for ix in registered:
        name=f"{arm}-{ix+1:05}";start=D/f"screen/inputs/{name}.json";end=D/f"screen/records/{name}.json"
        if not start.is_file() or not end.is_file():raise FileNotFoundError(name)
        x=json.loads(start.read_text());y=json.loads(end.read_text())
        assert x["id"]==y["id"]==name and x["index"]==ix
        row=dict(id=name,status=y["status"],steps=y.get("steps"))
        for label,tol in (("tight",.01),("loose",.1)):
            row["raw_"+label]=detect(x["k"],x["f"],x["z"],tol)
            row["relaxed_"+label]=detect(y["final_k"],y["final_f"],x["z"],tol) if "final_f" in y else 0
        rows.append(row)
    wanted={q:dict(Counter(x[q] for x in rows)) for q in ("raw_tight","relaxed_tight","raw_loose","relaxed_loose")}
    return dict(n=len(rows),counts=wanted,records=rows,
        exact_type_retained={q:sum(r["raw_"+q]>0 and r["raw_"+q]==r["relaxed_"+q] for r in rows) for q in ("tight","loose")})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arms",nargs="+");a=p.parse_args()
    dest=D/"catalog/post-relaxation-groups.json"
    out=json.loads(dest.read_text()) if dest.is_file() else {}
    for arm in a.arms:out[arm]=audit(arm);print(arm,out[arm]["exact_type_retained"])
    dest.write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
