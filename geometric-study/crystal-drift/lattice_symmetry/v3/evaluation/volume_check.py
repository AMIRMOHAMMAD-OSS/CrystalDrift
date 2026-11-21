"""Verify scale-gated sensitivity on all 480 ordered development draws using the public matcher."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from standard_match import SETTINGS,make_structure
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fit",default="small-angular-laplace-11")
    a=parser.parse_args();fit=V3/"models/fits"/a.fit
    chosen=json.loads((fit/"selection.json").read_text())["chosen"].split(".")[0]
    with np.load(fit/(chosen+"-dev.npz")) as f,np.load(V3/"models/fits/prior/dev.npz") as p:
        np.testing.assert_array_equal(f["ids"],p["ids"])
        ids=f["ids"];z=f["Z"];L=f["L"];F=f["F"];L0=p["L"]
    with np.load(ROOT/"scaled/mp20-data/val.npz") as truth:
        lut={str(x):i for i,x in enumerate(truth["ids"])}
        ref=[make_structure(truth["k"][lut[x]],truth["f"][lut[x]],truth["z"][lut[x]]) for x in ids]
    m=StructureMatcher(**SETTINGS);hits=[];rmsdiff=[];vraw=[];vcorrect=[]
    for i in range(len(ids)):
        for j in range(5):
            mask=z[i,j]>0;cell=L[i,j].astype(float)
            ratio=abs(np.linalg.det(L0[i,j])/np.linalg.det(cell))
            def check(l):
                q=Structure(l,z[i,j,mask],F[i,j,mask]);r=m.get_rms_dist(q,ref[i])
                return np.nan if r is None else r[0]
            a,b=check(cell),check(cell*np.cbrt(ratio))
            hits.append(np.isfinite(a)==np.isfinite(b));rmsdiff.append(abs(a-b))
            vraw.append(np.isfinite(a) and abs(np.log(abs(np.linalg.det(cell))/ref[i].volume))<=np.log(1.1))
            vcorrect.append(np.isfinite(b) and abs(np.log(abs(np.linalg.det(L0[i,j]))/ref[i].volume))<=np.log(1.1))
    out=dict(fit=fit.name,selected=chosen,draws=len(hits),unchanged_hit_flags=sum(map(int,hits)),
             max_normalized_rms_change=float(np.nanmax(rmsdiff)),native_targets_initial=int(np.reshape(vraw,(-1,5)).any(1).sum()),
             native_targets_rescaled=int(np.reshape(vcorrect,(-1,5)).any(1).sum()))
    (HERE/"volume-validation.json").write_text(json.dumps(out,indent=2));print(out)
if __name__=="__main__":main()
