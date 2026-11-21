"""Check the exact determinant law of the native centered-token Gaussian identity sampler."""
from pathlib import Path
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
def main():
    by_size={i:[] for i in range(1,21)}
    root=HERE/"predictions/prior-11/frozen/test"
    assert (root/"ready.json").exists()
    for p in sorted(root.glob("*.npz")):
        with np.load(p) as data:
            L=data["L"].astype(float)
            for sites in np.unique((data["Z"]>0).sum(1)):
                by_size[int(sites)].append(L[(data["Z"]>0).sum(1)==sites].reshape(-1,3,3))
    fits=[]
    for n,cells in by_size.items():
        if not cells:continue
        L=np.concatenate(cells);v=np.abs(np.linalg.det(L))
        var=1/n-.01
        expected=(2*np.sqrt(2/np.pi))*var**1.5
        # For a three-dimensional unit Gaussian matrix, E(det^2)=3!, by independence.
        se=np.sqrt(6-8/np.pi)*var**1.5/np.sqrt(len(L))
        fits.append(dict(sites=n,proposals=len(L),cell_component_variance=float(np.var(L)),expected_variance=var,
                         expected_volume=expected,observed_mean_volume=float(v.mean()),standard_error=se,
                         mean_volume_standardized_error=float((v.mean()-expected)/se),
                         volumes_below_point_one=int(np.sum(v<.1)),min_native=float(np.min(v))))
    assert sum(r["proposals"] for r in fits)==9046*20
    assert max(abs(r["mean_volume_standardized_error"]) for r in fits)<5
    receipt=dict(native_token_count=100,cohort="official test formulas only",ordering="all native zero-call draws",
                 proposals=sum(r["proposals"] for r in fits),below_point_one=sum(r["volumes_below_point_one"] for r in fits),
                 sitewise=fits)
    (HERE/"prior-law.json").write_text(json.dumps(receipt,indent=2)+"\n")
if __name__=="__main__":main()
