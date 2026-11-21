"""Separate fixed chart coordinates and locally isolated sites in actual development proposals."""
import json
import numpy as np
import torch
from .sampler import HERE,V3,ROOT,Sampler
from .network import pack
from lattice_symmetry.v2.neighbors import edges
torch.set_num_threads(1)

def main():
    s=Sampler();fold=HERE/'fits/small-angular-laplace-11';model=s.load(fold/json.loads((fold/'selection.json').read_text())['chosen'])
    ids=json.loads((V3/'protocol/splits.json').read_text())['dev_ids']
    with np.load(ROOT/'scaled/mp20-data/val.npz') as q:zs=q['z'];ii={str(x):i for i,x in enumerate(q['ids'])}
    records=[]
    for mid in ids:
        k=ii[mid];seed=int(np.random.SeedSequence([38471,k]).generate_state(1)[0]);seeds=s.seeds(zs[k],5,seed)
        x=pack(seeds);aa,bb,_,_=edges(*x[:3],4.)
        b,n=x[2].shape
        cross=torch.bincount(aa[aa!=bb],minlength=b*n).reshape(b,n)
        with torch.inference_mode():cell,frac=model(*x)
        moved=(((frac-x[1]+.5)%1-.5)@x[0]).norm(dim=-1)
        for j,r in enumerate(seeds):
            for site in range(len(r.Z)):
                records.append((mid,j,site,len(r.Z),int(x[5][j,site]),int(cross[j,site]),float(moved[j,site])))
    t=np.array([r[3:] for r in records]);result={}
    for key,mask in [('all',np.ones(len(t),bool)),('9plus',t[:,0]>=9)]:
        a=t[mask];out={}
        for name,m in [('fixed',a[:,1]==0),('free_without_distinct_neighbor',(a[:,1]>0)&(a[:,2]==0)),
                       ('free_with_distinct_neighbor',(a[:,1]>0)&(a[:,2]>0))]:
            out[name]=dict(sites=int(m.sum()),zero_at_1e_5_A=int((a[m,3]<1e-5).sum()),
                           median_A=float(np.median(a[m,3])) if m.any() else None,p90_A=float(np.quantile(a[m,3],.9)) if m.any() else None)
        result[key]=out
    (HERE/'site-limitations.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(HERE/'site-limitations.npz',id=[r[0] for r in records],draw=[r[1] for r in records],
        site=[r[2] for r in records],n=t[:,0].astype(int),free=t[:,1].astype(int),distinct_edges=t[:,2].astype(int),step_A=t[:,3])
    print(result,flush=True)
if __name__=='__main__':main()

