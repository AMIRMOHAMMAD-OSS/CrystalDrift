"""Seven-class auxiliary supervision for one same-backbone training-only control."""
import json
from pathlib import Path
import numpy as np
import torch
from .sampler import HERE,V3

SYSTEM_NAMES=('triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic')
SYSTEM_UPPER=np.array([2,15,74,142,167,194,230])
COEFFICIENT=.1

def labels(eligible):
    protected=set(json.loads((V3/'protocol/splits.json').read_text())['train_ids'])
    groups={}
    for fn in sorted((V3/'uflow/train-cache').glob('*.npz')):
        with np.load(fn) as record:
            for mid,sg in zip(record['ids'].tolist(),record['sg'].tolist()):
                if mid not in protected or mid in groups or not 1<=sg<=230:raise ValueError('Invalid protected training group')
                groups[mid]=sg
    masked=[groups[mid] for mid in eligible]
    out=np.searchsorted(SYSTEM_UPPER,np.array(masked))
    return out

def loss(head,pool,n,classes,pidx,device):
    # The two active differentiable draws share the exact formula of both observed positives.
    # Their partner IDs/index order are the same as the paired continuous training targets.
    rows=pool.reshape(n,4,-1)[:,:2].reshape(n*2,-1)
    targets=torch.as_tensor(classes[pidx],dtype=torch.long,device=device)
    logits=head(rows)
    return torch.nn.functional.cross_entropy(logits,targets),logits,targets
