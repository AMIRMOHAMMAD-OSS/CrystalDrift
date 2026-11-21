"""A/B verification and timings for the integer neighbor-enumeration training kernel."""
import hashlib,json,os,sys,time
from pathlib import Path
if os.getenv('CRYTALDRIFT_NUMBA_SITE'):sys.path.append(os.environ['CRYTALDRIFT_NUMBA_SITE'])
import numpy as np
from lattice_symmetry.v2.neighbors import fast_indices
from .neighbor_kernel import scalar_indices
from .sampler import HERE

def main():
    files=sorted((HERE/'cache').glob('train-*.npz'))
    digest=hashlib.sha256();elapsed=[0.,0.];n=0;edges=0
    def check(L,F,Z,cut=4.):
        nonlocal n,edges
        args=(np.ascontiguousarray(L,dtype=np.float64),np.ascontiguousarray(F,dtype=np.float64),
              np.ascontiguousarray(Z,dtype=np.int64),np.linalg.inv(L.astype(np.float64)),cut)
        found=[]
        for k,impl in enumerate((fast_indices,scalar_indices)):
            stamp=time.perf_counter();found.append(impl(*args));elapsed[k]+=time.perf_counter()-stamp
        if not np.array_equal(*found):raise RuntimeError('integer indices/order differ')
        digest.update(found[0].tobytes());n+=len(L);edges+=len(found[0])
    # All held cartesian lengths are on, just below, or above the physical sphere.
    for cutoff in (4.,np.nextafter(4.,0.),np.nextafter(4.,5.)):
        check(np.tile(np.eye(3)*4.,(2,1,1)),np.array([[[0.,0.,0.],[.5,.5,0.]]]*2),
              np.array([[4,4],[4,0]]),cutoff)
    elapsed[:]=[0.,0.]
    for fn in files:
        with np.load(fn) as b:
            l,f,z=(b['x'+str(i)] for i in range(3))
        # Includes the identical cell set used in a complete training traversal.
        for a in range(0,len(l),64):check(l[a:a+64],f[a:a+64],z[a:a+64])
    record=dict(geometries=n,directed_edges=edges,identical_order=True,sha256=digest.hexdigest(),
                old_cpu_seconds=elapsed[0],scalar_cpu_seconds=elapsed[1],num_shards=len(files),
                scope='all first-traversal seed cells; ordered edges and cutoff boundary fixtures')
    (HERE/'neighbor-kernel-parity.json').write_text(json.dumps(record,indent=2))
    print(record)

if __name__=='__main__':main()
