"""Independently regenerate and hash each pre-network latent from the historical CPU RNGs."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch

HERE=Path(__file__).resolve().parent

def hashes(generator,n):
    raw=torch.randn((n,100,12),generator=generator)
    centered=raw-raw.mean(dim=1,keepdim=True)
    return np.array([hashlib.sha256(memoryview(z.numpy()).cast('B')).hexdigest() for z in centered],dtype='S64')

def run(branch):
    torch.set_num_threads(2)
    checked=0
    if branch=='prefix':
        assert torch.__version__=='2.7.1+cpu'
        with np.load(HERE/'published-prefix.npz') as d:expected=d['per_input_sha'].ravel().astype('S64')
        g=torch.Generator(device='cpu').manual_seed(2026)
        for i in range(0,len(expected),12):
            tgt=expected[i:i+12]
            np.testing.assert_array_equal(hashes(g,len(tgt)),tgt);checked+=len(tgt)
        assert checked==640
    else:
        assert torch.__version__=='2.6.0+cu124'
        pool=HERE/'predictions/published-prior/frozen/test'
        for p in sorted(pool.glob('?????-?????.npz')):
            lo,hi=map(int,p.stem.split('-'))
            with np.load(p) as d:recorded=d['per_input_sha']
            for j in range(5 if lo==0 else 0,20):
                for i in range(lo,hi,24):
                    g=torch.Generator(device='cpu').manual_seed(2026000000+1000000*j+i)
                    n=min(hi-i,24)
                    np.testing.assert_array_equal(hashes(g,n),recorded[i-lo:i-lo+n,j]);checked+=n
        assert checked==180280
    info=dict(branch=branch,torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
              platform=platform.platform(),separately_recoded_latents_verified=checked)
    (HERE/f'published-{branch}-identity.json').write_text(json.dumps(info,indent=2)+'\n')
    print(info,flush=True)

if __name__=='__main__':
    p=ArgumentParser();p.add_argument('branch',choices=['prefix','remainder']);run(p.parse_args().branch)
