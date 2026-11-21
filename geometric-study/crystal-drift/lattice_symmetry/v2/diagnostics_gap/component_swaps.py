"""Localize the source of the one-call recovery gap using chart-dependent cell/site swaps."""
import csv
import json
import os
from pathlib import Path
import sys
for thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread]='1'
import numpy as np
from importlib.metadata import version
out=Path(__file__).resolve().parent
v2=out.parent
root=v2.parents[1]
sys.path.insert(0,str(root/'revision/full'))
from match_full import candidate,target_context,unpack_gram
assert version('pymatgen')=='2024.4.13'
num=512
with np.load(root/'revision/full/mp20/cohort.npz') as d: ref={k:d[k][:num] for k in d.files}
# Draw zero from the established, hash-ordered cohort. Keep both model seed and file order fixed.
uk=[]; uf=[]; uids=[]; uz=[]
for p in sorted((root/'revision/full/external/mp20').glob('draw00-cases*.npz')):
    with np.load(p) as d:
        uk.extend(d['k']); uf.extend(d['f']); uids.extend(d['ids'].tolist()); uz.extend(d['z'])
    if len(uk)>=num:break
np.testing.assert_array_equal(uids[:num],ref['ids']);np.testing.assert_array_equal(uz[:num],ref['z'])
uk=np.asarray(uk[:num]); uf=np.asarray(uf[:num])
folder=v2/'csp/test/angular-seed11'
ak=np.load(folder/'k.npy',mmap_mode='r')[:num,0]; af=np.load(folder/'f.npy',mmap_mode='r')[:num,0]
with np.load(folder/'outcomes.npz') as d: archived_a=d['rms'][:num,0]; np.testing.assert_array_equal(d['ids'][:num],ref['ids'])
with np.load(root/'revision/full/external/mp20/complete-crosswalk-k20.npz') as d: archived_u=d['rms'][:num,0]
names=('angular_cell_angular_sites','uflow_cell_angular_sites','angular_cell_uflow_sites','uflow_cell_uflow_sites','reference_cell_angular_sites','angular_cell_reference_sites')
rs=np.full((num,len(names)),np.nan)
for i in range(num):
    z0=ref['z'][i];z=z0[z0>0].astype(np.int64);ctx=target_context(ref['k'][i],ref['f'][i],z0)
    klist=(ak[i],uk[i],ak[i],uk[i],ref['k'][i],ak[i]);flist=(af[i],af[i],uf[i],uf[i],af[i],ref['f'][i])
    for j,(k,f) in enumerate(zip(klist,flist)):
        rs[i,j]=candidate(ctx,k,f,z,unpack_gram(k),i%199==0)[0]
np.testing.assert_allclose(rs[:,0],archived_a,equal_nan=True,atol=1e-7,rtol=1e-5)
np.testing.assert_allclose(rs[:,3],archived_u,equal_nan=True,atol=1e-7,rtol=1e-5)
with (out/'component_swaps.csv').open('w',newline='') as fp:
    w=csv.writer(fp);w.writerow(['id','sites',*names]);w.writerows((mid,int((z>0).sum()),*rr) for mid,z,rr in zip(ref['ids'],ref['z'],rs))
n=(ref['z']>0).sum(1)
outcomes={key:{'n':int(ix.sum()), 'hits':dict(zip(names,np.isfinite(rs[ix]).sum(0).tolist()))} for key, ix in {'all':np.ones(num,dtype=bool),'1_4':n<=4,'5_8':(n>4)&(n<=8),'9_12':(n>8)&(n<=12),'13_20':n>12}.items()}
(out/'component_swaps.json').write_text(json.dumps(outcomes,indent=2))
print(json.dumps(outcomes))
