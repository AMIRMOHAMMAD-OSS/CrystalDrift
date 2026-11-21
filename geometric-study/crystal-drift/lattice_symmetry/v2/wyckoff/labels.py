"""Extract the labelled conventional structures and verify each round trip on official splits."""
import argparse
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import spglib
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from charts import HERE,expand,label

ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from data import unpack_gram
from standard_match import SETTINGS

@lru_cache(None)
def dataset(split):
    with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as d:
        return d["ids"],np.linalg.cholesky(unpack_gram(d["k"])),d["f"],d["z"]

def chunk(job):
    split,first,last=job
    path=HERE/f"labels/{split}-{first:05}.jsonl.gz"
    if path.exists():return
    ids,cells,fs,zs=dataset(split)
    matcher=StructureMatcher(**SETTINGS)
    temp=path.with_suffix(".part")
    with gzip.open(temp,"wt",encoding="utf8") as out:
        for i in range(first,last):
            mask=zs[i]>0
            z=zs[i,mask];f=fs[i,mask];cell=cells[i]
            row=dict(id=str(ids[i]),index=i)
            try:
                r=label(cell,f,z)
                rc=np.array(r["cell"]);rf,rz=expand(r["group"],r["template"],r["u"])
                # Conventional expansion vs input, primitive matching with native scale.
                a=Structure(cell,z,f); b=Structure(rc,rz,rf)
                rms=matcher.get_rms_dist(a,b)
                if rms is None: raise ValueError("StructureMatcher-roundtrip")
                row.update(r,roundtrip_rms=float(rms[0]))
            except Exception as e:
                row["error"]=str(e)
            out.write(json.dumps(row,separators=(",",":"))+"\n")
    temp.replace(path)

def read(split):
    for p in sorted((HERE/"labels").glob(split+"-*.jsonl.gz")):
        with gzip.open(p,"rt",encoding="utf8") as fh:
            yield from map(json.loads,fh)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=3);p.add_argument("--limit",type=int)
    p.add_argument("--splits",nargs="+",default=["train","test","val"]);p.add_argument("--pilot",action="store_true");args=p.parse_args()
    (HERE/"labels").mkdir(exist_ok=True)
    jobs=[]
    for s in args.splits:
        n=len(dataset(s)[0]);jobs.extend((s,i,min(n,i+200)) for i in range(0,min(n,args.limit or n),200))
    t=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j,_ in enumerate(pool.map(chunk,jobs)):
            if j%10==0:print(j+1,len(jobs),round(perf_counter()-t,1),flush=True)
