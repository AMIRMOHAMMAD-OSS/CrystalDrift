"""Four radial shells for the secondary independent-resolution screen."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from .sampler import HERE,Sampler
from .network import pack
from .prepare import CACHE,safe_save
from ..geometry.descriptors import dense_pair_shells,fit_stats
torch.set_num_threads(1)

def main():
    s=Sampler();items=[]
    for first in range(0,len(s.rows),48):
        sample=[s.target(i) for i in range(first,min(first+48,len(s.rows)))]
        L,F,Z,*_=pack(sample)
        with torch.no_grad():items.append(dense_pair_shells(L,F,Z).numpy())
    values=np.concatenate(items)
    with np.load(CACHE/'calibration.npz') as raw:
        ix=[s.mid[str(k)] for k in raw['ids']]
    mean,scale=fit_stats(dict(radial4=values[ix]))['radial4']
    safe_save(CACHE/'radial4.npz',values=values,mean=mean,scale=scale,
              ids=np.array(s.ids),calibration=np.array(ix))
    print(values.shape,flush=True)
if __name__=='__main__':main()

