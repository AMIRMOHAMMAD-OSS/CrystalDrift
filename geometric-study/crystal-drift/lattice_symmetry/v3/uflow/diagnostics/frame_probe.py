"""Identify public-check and lattice-enumeration costs on a native prior shard."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE.parent/"evaluation"))
from score import Lattice,target_context,truth,frame_possible

def main(a):
    lookup,targets=truth("test")
    rows=[]
    with np.load(a.source) as d:
        ids,Ls,Fs,Z=(d[k] for k in ("ids","L","F","Z"))
    for i in range(a.first,a.last):
        ix=lookup[str(ids[i])]
        ctx=target_context(targets['k'][ix],targets['f'][ix],targets['z'][ix])
        _,ref,ref_f,ref_z,anchor,ref_g=ctx
        z=Z[i][Z[i]>0].astype(np.int64)
        for j,L in enumerate(Ls[i].astype(float)):
            cond=np.linalg.cond(L)
            if cond<200:continue
            A=Lattice(L)
            U=np.rint(A.lll_mapping).astype(np.int64)
            M=U@L
            f=(Fs[i,j,:len(z)]@np.linalg.inv(U))%1
            forced=(ix*104729+j*1543)%3001==0
            ok=(len(ref)!=len(z) or np.linalg.cond(M)>=1000 or
                frame_possible(M@M.T,np.ascontiguousarray(f),z,ref_g,ref_f,ref_z,anchor))
            if ok or forced:
                rows.append(dict(id=str(ids[i]),row=i,draw=j,volume=float(abs(np.linalg.det(L))),condition=float(cond),
                     condition_lll=float(np.linalg.cond(M)),reduced_n=len(ref),n=len(z),forced=forced,frame_ok=bool(ok)))
    print(json.dumps(rows,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("first",type=int);p.add_argument("last",type=int)
    main(p.parse_args())
