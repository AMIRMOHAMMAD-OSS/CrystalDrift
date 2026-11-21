import csv
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

root = Path("geometric-study/crystal-drift").resolve()
folder = Path(__file__).resolve().parent
sys.path.insert(0, str(root))


def score(job):
    method, seed = job
    from importlib.metadata import version
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import data
    from standard_match import SETTINGS, make_structure, paths, order

    assert version("pymatgen") == "2024.4.13"
    old = json.loads((root/"structure-matching"/f"multi-{method}-{seed}.json").read_text())
    test = np.load(root/"data/multi-test.npz")
    ix = order("multi",test["ids"])
    np.testing.assert_array_equal(test["ids"][ix],old["material_ids"])
    if method == "retrieval":
        with np.load(root/"data/multi-train.npz") as train:
            k,f,_ = data.retrieval_baseline(test,train,20,seed)
        k,f = k[ix],f[ix]
    else:
        with np.load(paths("multi",method,seed)) as src:
            np.testing.assert_array_equal(src["ids"],test["ids"])
            np.testing.assert_array_equal(src["z"],test["z"])
            k,f = src["k"][ix],src["f"][ix]
    assert k.shape[:2] == (128,20)
    z,kt,ft = (test[key][ix] for key in ("z","k","f"))
    matcher = StructureMatcher(**SETTINGS)
    distances = np.full((128,20),np.nan)
    failures=0
    for i in range(128):
        ref=make_structure(kt[i],ft[i],z[i])
        cache={}
        for j in range(20):
            key=(k[i,j].tobytes(),f[i,j].tobytes())
            if key not in cache:
                try:
                    pred=make_structure(k[i,j],f[i,j],z[i])
                    value=matcher.get_rms_dist(pred,ref)
                    cache[key]=float(value[0]) if value is not None else np.nan
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):
                    failures+=1
                    cache[key]=np.nan
            distances[i,j]=cache[key]
    np.testing.assert_allclose(distances[:,:5],np.array(old["rms_by_candidate"],float),equal_nan=True,atol=1e-7,rtol=1e-7)
    ok=np.isfinite(distances).any(axis=1)
    out=dict(method=method,seed=seed,n=128,k=20,matched=int(ok.sum()),mr=ok.mean(),
             rmse=np.nanmin(distances[ok],axis=1).mean(),exception_count=failures)
    np.savez_compressed(folder/f"match20-{method}-{seed}.npz",ids=test["ids"][ix],rms=distances)
    (folder/f"match20-{method}-{seed}.json").write_text(json.dumps(out),encoding="utf8")
    print(method,seed,out["matched"],round(out["rmse"],5),flush=True)
    return out


if __name__=="__main__":
    jobs=[(m,s) for m in ("exact","flow1","flow16","retrieval") for s in (11,23,37)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        records=list(executor.map(score,jobs))
    (folder/"match20.json").write_text(json.dumps(records,indent=2),encoding="utf8")
