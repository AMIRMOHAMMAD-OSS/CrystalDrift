from pathlib import Path
import sys, json
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
def score(job):
    root,method=job
    sys.path.insert(0,str(root))
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from standard_match import SETTINGS,make_structure
    base=root/'scaled'
    run=base/'external'/'wbm' if method=='uflow' else base/'runs'/'wbm'
    seed=2026 if method=='uflow' else 11
    stem=f'{method}-seed{seed}-test'
    known=json.loads((run/(stem+'-matching.json')).read_text())
    old=np.isfinite(np.array(known['rms_by_candidate'],float))
    with np.load(run/(stem+'.npz')) as a:ids,z,k,f=[a[x] for x in ('ids','z','k','f')]
    with np.load(base/'wbm-data/test.npz') as b: lookup={v:i for i,v in enumerate(b['ids'])};ref=[b[x][[lookup[v] for v in ids]] for x in ('z','k','f')]
    hits=[];errs=Counter();mat=StructureMatcher(**SETTINGS)
    for i in range(len(ids)):
        truth=make_structure(ref[1][i],ref[2][i],ref[0][i])
        mem={};r=[]
        for j in range(5):
            key=(k[i,j].tobytes(),f[i,j].tobytes())
            if key not in mem:
                try:mem[key]=bool(mat.fit(make_structure(k[i,j],f[i,j],z[i]),truth))
                except (ValueError, FloatingPointError,np.linalg.LinAlgError,OverflowError) as e:
                    errs[type(e).__name__]+=1;mem[key]=False
            r.append(mem[key])
        hits.append(r)
    new=np.array(hits)
    return dict(method=method,seed=seed,cases=len(ids),rms5=int(old.any(1).sum()),strict5=int(new.any(1).sum()),strict1=int(new[:,0].sum()),strict_only=int((new&~old).sum()),old_only_candidates=int((old&~new).sum()),exceptions=dict(errs))
if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    with ProcessPoolExecutor(max_workers=3) as ex:
        out=list(ex.map(score,[(root,m) for m in ('drift','flow1','flow16','retrieval','uflow')]))
    print(json.dumps(out,indent=1))
    Path(__file__).with_name('strict-wbm.json').write_text(json.dumps(out,indent=1))

