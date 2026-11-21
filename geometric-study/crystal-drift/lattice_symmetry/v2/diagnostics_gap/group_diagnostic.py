"""Check spontaneous symmetry in formula-only predictions on the same 512 targets."""
import csv
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import spglib
out=Path(__file__).resolve().parent
v2=out.parent
root=v2.parents[1]
sys.path.insert(0,str(root))
from data import unpack_gram
n=512
with np.load(root/'revision/full/mp20/cohort.npz') as d:ref={k:d[k][:n] for k in d.files}
us=[np.load(p) for p in sorted((root/'revision/full/external/mp20').glob('draw00-cases*.npz'))[:3]]
uk,uf,ids=[np.concatenate([d[k] for d in us])[:n] for k in ['k','f','ids']]
np.testing.assert_array_equal(ids,ref['ids'])
for d in us:d.close()
folder=v2/'csp/test/angular-seed11'
ak=np.load(folder/'k.npy',mmap_mode='r')[:n,0];af=np.load(folder/'f.npy',mmap_mode='r')[:n,0]
def number(k,f,z,eps):
 mask=z>0
 dataset=spglib.get_symmetry_dataset((np.linalg.cholesky(unpack_gram(k)),f[mask],z[mask]),symprec=eps)
 if dataset is None:return 0
 return int(dataset['number'])
with (out/'group_diagnostic.csv').open('w',newline='') as fp:
 w=csv.writer(fp);w.writerow(['symprec','id','sites','reference_group','angular_group','uflow_group'])
 for eps in [.01,.1]:
  rs=[]
  for i in range(n):
   z=ref['z'][i];g=np.array([number(k,f,z,eps) for k,f in [(ref['k'][i],ref['f'][i]),(ak[i],af[i]),(uk[i],uf[i])]])
   rs.append(g)
   w.writerow([eps,ids[i],sum(z>0),*g])
  r,a,u=np.array(rs).T
  for label,b in [('ang',a),('uflow',u)]:print(eps,label,'n',len(b),'P1',sum(b==1),'matches',sum((r==b)&(b>0)),'failed',sum(b==0),'referenceP1',sum(r==1))
