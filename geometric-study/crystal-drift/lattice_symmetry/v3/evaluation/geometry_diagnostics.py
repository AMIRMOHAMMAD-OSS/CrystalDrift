"""Native shape and coupled site-motion diagnostics on consistently represented structures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from pymatgen.core import Lattice
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from data import unpack_gram
from scipy.linalg import eigvalsh
from storage import pool

def shape(L):
    R=Lattice(L).get_niggli_reduced_lattice().matrix
    G=R@R.T
    return G/np.cbrt(np.linalg.det(G))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("panel");ap.add_argument("methods",nargs="+");ap.add_argument("--limit",type=int,default=96)
    a=ap.parse_args();subset=json.loads((V3/"protocol/splits.json").read_text())[a.panel+"_ids"][:a.limit]
    split="test" if a.panel=="test" else "val" if a.panel in ("confirm","dev") else "train"
    with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as ds:
        lookup={s:i for i,s in enumerate(ds["ids"])}
        target={s:shape(np.linalg.cholesky(unpack_gram(ds["k"][lookup[s]]))) for s in subset}
    for m in a.methods:
        records=[]
        for file in sorted(pool(a.panel,m).glob("?????-?????.npz")):
            with np.load(file) as x,np.load(pool(a.panel,"prior")/file.name) as p:
                np.testing.assert_array_equal(x["ids"],p["ids"]);ids=x["ids"]
                if not any(s in target for s in ids):continue
                L,F,Z=x["L"],x["F"],x["Z"];L0,F0,Z0=p["L"],p["F"],p["Z"]
            for i,s in enumerate(ids):
                if s not in target:continue
                for j in range(L.shape[1]):
                    active=Z[i,j]>0
                    if not active.any() or not np.isfinite(L[i,j]).all():continue
                    assert np.array_equal(Z[i,j],Z0[i,j])
                    delta=F[i,j,active]-F0[i,j,active]
                    disp=np.linalg.norm((delta-np.rint(delta))@L0[i,j],axis=1)
                    G=shape(L[i,j])
                    # Sorted log eigenvalues remove residual signed-permutation ties after reduction.
                    sh=np.linalg.norm(np.log(eigvalsh(G))-np.log(eigvalsh(target[s])))
                    records.append((s,j+1,sh,np.median(disp),disp.max(),np.log(abs(np.linalg.det(L[i,j])/np.linalg.det(L0[i,j])))))
        arr=np.array(records,dtype=[("id","U20"),("draw","i2"),("shape","f8"),("site_median","f8"),("site_max","f8"),("delta_volume","f8")])
        dst=HERE/"diagnostics";dst.mkdir(exist_ok=True)
        np.savez_compressed(dst/f"{a.panel}-{m}.npz",draws=arr,ids=subset)
        print(m,len(arr),{k:float(np.median(arr[k])) for k in ("shape","site_median","delta_volume")} if len(arr) else {})
if __name__=="__main__":main()
