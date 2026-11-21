"""Public matcher check on the observed CPU/CUDA development disagreement."""
import json
from importlib.metadata import version
import sys
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from data import unpack_gram
from standard_match import SETTINGS

def main():
    target='mp-8584';draw=1
    with np.load(ROOT/'scaled/mp20-data/val.npz') as data:
        i=data['ids'].tolist().index(target);z=data['z'][i];keep=z>0
        ref=Structure(np.linalg.cholesky(unpack_gram(data['k'][i])),z[keep],data['f'][i,keep])
    records={}
    for name,source in [('cuda',HERE/'fits/small-angular-laplace-11/half-dev.npz'),('cpu',HERE/'controls/dev-inputs-cpu.npz')]:
        with np.load(source) as q:
            i=q['ids'].tolist().index(target);L=q['L'][i,draw];F=q['F'][i,draw];Z=q['Z'][i,draw]
            proposal=Structure(L,Z[Z>0],F[Z>0])
        with np.load(source.with_name(source.stem+'-scores.npz')) as q:
            stored=float(q['rms'][i,draw])
        matcher=StructureMatcher(**SETTINGS)
        rms=matcher.get_rms_dist(proposal,ref)
        passed=rms is not None
        records[name]=dict(stock_rms=None if rms is None else list(map(float,rms)),matches_public=passed,
                            stored_rms=stored if np.isfinite(stored) else None,agrees=bool(passed==np.isfinite(stored)))
    payload=dict(id=target,draw=draw+1,reference_n=len(ref),pymatgen=version('pymatgen'),platform=sys.platform,sources=records)
    (HERE/'boundary-matcher.json').write_text(json.dumps(payload,indent=2));print(payload,flush=True)
if __name__=='__main__':main()

