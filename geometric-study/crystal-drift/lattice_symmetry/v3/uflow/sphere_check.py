"""Compare all raw image labels and distances for the unmodified and requested-radius public search."""
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"evaluation"))
from score import MATCHER, Lattice, Structure, target_context, tiny_sphere_search, truth

def ordered(value):
    f,d,i,img=value
    if len(i)==0:
        return np.zeros((0,5))
    mat=np.column_stack((i,img,d))
    return mat[np.lexsort((mat[:,3],mat[:,2],mat[:,1],mat[:,0]))]

def main():
    rng=np.random.default_rng(2762)
    rows=[]
    for j in range(80):
        a=np.diag(rng.uniform(.12,.7,3))
        a[0,1]=rng.uniform(-.8,.8);a[1,2]=rng.uniform(-.2,.2)
        lat=Lattice(a)
        points=rng.random((1+j%3,3))
        center=rng.uniform(-.6,.6,3);r=rng.uniform(.06,.78)
        stock=ordered(lat.get_points_in_sphere(points,center,r,zip_results=False))
        with tiny_sphere_search():
            actual=ordered(lat.get_points_in_sphere(points,center,r,zip_results=False))
        np.testing.assert_allclose(actual,stock,rtol=1e-11,atol=1e-11)
        rows.append(dict(radius=float(r),condition=float(np.linalg.cond(a)),images=len(actual),
                         max_error=float(np.max(np.abs(stock-actual))) if len(actual) else 0.))
    _,ref=truth("val")
    one=int(np.flatnonzero((ref["z"]>0).sum(1)==1)[0])
    ctx=target_context(ref["k"][one],ref["f"][one],ref["z"][one])
    extreme=[]
    for L in (np.eye(3)*.001,):
        pred=Structure(L,[int(ref["z"][one,0])],[[0,0,0]])
        start=time.perf_counter()
        with tiny_sphere_search():
            hit=MATCHER.get_rms_dist(pred,ctx[0]);strict=MATCHER.fit(pred,ctx[0])
        extreme.append(dict(volume=float(abs(np.linalg.det(L))),condition=float(np.linalg.cond(L)),
                       seconds=time.perf_counter()-start,matched=bool(hit is not None),strict=bool(strict)))
    (HERE/"sphere-search-parity.json").write_text(json.dumps(dict(calls=rows,tiny_cases=extreme),indent=2)+"\n")
if __name__=="__main__":main()
