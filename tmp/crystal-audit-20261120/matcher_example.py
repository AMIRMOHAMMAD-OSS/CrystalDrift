import sys,json
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
from standard_match import SETTINGS, make_structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from importlib.metadata import version
folder=root/'scaled/runs/wbm'
pred=np.load(folder/'drift-seed11-test.npz'); src=np.load(root/'scaled/wbm-data/test.npz')
lookup={k:i for i,k in enumerate(src['ids'])}; ids,z,k,f=[pred[i] for i in ('ids','z','k','f')]; rk,rf,rz=[src[i] for i in ('k','f','z')]
sc=np.array(json.loads((folder/'drift-seed11-test-matching.json').read_text())['rms_by_candidate'],dtype=float)
mat=StructureMatcher(**SETTINGS);checked=0
for i,j in np.argwhere(np.isfinite(sc)):
    idx=lookup[ids[i]]; a=make_structure(k[i,j],f[i,j],z[i]); b=make_structure(rk[idx],rf[idx],rz[idx]); r=mat.get_rms_dist(a,b);assert r is not None and abs(r[0]-sc[i,j])<1e-5
    checked+=1
    if not mat.fit(a,b):
        print(json.dumps(dict(version=version('pymatgen'),ID=str(ids[i]),draw=int(j+1),rms=r[0],max_site=r[1],fit=False,checked=checked)));break
else:print('all inspected hits agreed',checked)

