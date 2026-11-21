"""Reconstruct fixed development input tensors from archived prior arrays and their orbit provenance."""
import json
import numpy as np
from .sampler import HERE,V3,Seed,orbit_chart
from .network import pack
from .prepare import safe_save,CACHE

def main():
    raw=V3/'evaluation/raw/dev/prior/00000-00096.npz'
    prior=HERE/'fits/prior/dev.npz'
    with np.load(raw) as meta,np.load(prior) as p:
        assert np.array_equal(meta['ids'],p['ids'])
        groups=meta['group'][:,:5];patterns=[json.loads(row)[:5] for row in meta['patterns']]
        L,F,Z=p['L'],p['F'],p['Z']
        assert np.array_equal(Z,meta['Z'][:,:5])
        assert np.max(np.abs(L-meta['L'][:,:5]))<1e-6
        assert np.max(np.abs(F[Z>0]-meta['F'][:,:5][Z>0]))<1e-6
        seeds=[]
        for i in range(len(groups)):
            for j,g in enumerate(groups[i]):
                pat=tuple(tuple(v) for v in patterns[i][j]);inds=[];jac=[];zs=[]
                for orbit,(el,letter) in enumerate(pat):
                    J,_=orbit_chart(g,letter);inds.extend([orbit]*len(J));jac.extend(J);zs.extend([el]*len(J))
                n=len(zs);assert np.array_equal(Z[i,j,:n],zs)
                seeds.append(Seed(L[i,j],F[i,j,:n],np.array(zs),np.array(inds),np.array(jac),g,pat,'',''))
        args=pack(seeds)
        safe_save(CACHE/'dev-inputs.npz',ids=p['ids'],**{'x'+str(i):a.numpy() for i,a in enumerate(args)})
if __name__=='__main__':main()

