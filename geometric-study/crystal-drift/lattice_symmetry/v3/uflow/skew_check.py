"""Compare every origin image, its order, and every distance against the pinned routine."""
import json
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'evaluation'))
from score import origin_images,tiny_sphere_search,truth,target_context,Structure,MATCHER
from pymatgen.optimization.neighbors import find_points_in_spheres

def main():
    rng=np.random.default_rng(31722)
    cases=images=0;err=0.
    for i in range(100):
        L=np.diag(rng.uniform(.18,1.6,3))
        U=np.eye(3);U[0,1]=rng.integers(-13,14);U[2,1]=rng.integers(-8,9)
        kw=dict(all_coords=np.zeros((1,3)),center_coords=np.zeros((1,3)),lattice=U@L,
                r=rng.uniform(.05,2.2),pbc=np.ones(3,dtype=np.int_),tol=1e-8)
        a=find_points_in_spheres(**kw);b=origin_images(find_points_in_spheres,kw,minimum_condition=0.)
        if b is None:continue
        for h in range(3):np.testing.assert_array_equal(a[h],b[h])
        np.testing.assert_allclose(a[3],b[3],atol=1e-12,rtol=1e-12)
        images+=len(a[3]);cases+=1;err=max(err,np.max(np.abs(a[3]-b[3])))
    lookup,target=truth('test')
    with np.load(HERE/'predictions/prior-11/frozen/test/01536-01792.npz') as data:
        i,j=138,18;mid=str(data['ids'][i]);idx=lookup[mid];L=data['L'][i,j].astype(float);f=data['F'][i,j];z=data['Z'][i];mask=z>0
        ctx=target_context(target['k'][idx],target['f'][idx],target['z'][idx]);candidate=Structure(L,z[mask],f[mask]);before=time.perf_counter()
        with tiny_sphere_search():
            rms=MATCHER.get_rms_dist(candidate,ctx[0]);strict=MATCHER.fit(candidate,ctx[0])
    assert cases>=90
    out=dict(fixtures=cases,images=images,order_and_labels_exact=True,max_distance_error=float(err),
             actual=dict(id=mid,row=i,draw=j,condition=float(np.linalg.cond(L)),determinant=float(np.linalg.det(L)),
                         rms=None if rms is None else float(rms[0]),strict=bool(strict),seconds=time.perf_counter()-before))
    (HERE/'skew-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
