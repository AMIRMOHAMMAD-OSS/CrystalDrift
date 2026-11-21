"""Measure numerical reproducibility of one retained training bank on the current CPU."""
import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
if os.environ.get("CRYTALDRIFT_NUMBA_SITE"):
    sys.path.append(os.environ["CRYTALDRIFT_NUMBA_SITE"])
import numpy as np
import torch
from lattice_symmetry.v3.models.prepare import BLOCK, CACHE, make_shard
from lattice_symmetry.v3.models.sampler import Sampler


def main(output):
    from lattice_symmetry.v2 import neighbors
    neighbors.fast_indices = None
    epoch, shard = 1, 0
    service = Sampler('cpu')
    order = np.random.default_rng(55171+epoch).permutation(len(service.rows))
    candidate = make_shard(service,order[:BLOCK],shard,epoch,in_memory=True)
    stream = io.BytesIO()
    np.savez_compressed(stream,**candidate)
    original = CACHE/'epoch-01/train-000.npz'
    fields = {}
    with np.load(original) as bank:
        assert set(bank.files) == set(candidate)
        for key, value in candidate.items():
            actual = np.asarray(value)
            expected = bank[key]
            assert actual.shape == expected.shape
            fields[key] = dict(source_dtype=str(expected.dtype),replay_dtype=str(actual.dtype),
                               differing_values=int(np.count_nonzero(actual!=expected)),
                               max_abs_error=float(np.max(np.abs(actual.astype(float)-expected.astype(float)))))
    record = dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                  regenerated_sha256=hashlib.sha256(stream.getvalue()).hexdigest(),
                  python=sys.version,numpy=np.__version__,torch=torch.__version__,neighbor_indices='NumPy fallback',fields=fields)
    output.write_text(json.dumps(record,indent=2)+'\n')

def exact_later(output):
    import numba
    registry=json.loads((CACHE.parent/'fresh-bank-sha256.json').read_text())['shards']
    service=Sampler('cpu')
    checks={}
    for epoch,shard in ((1,194),(7,206)):
        order=np.random.default_rng(55171+epoch).permutation(len(service.rows))
        data=make_shard(service,order[shard*BLOCK:(shard+1)*BLOCK],shard,epoch,in_memory=True)
        stream=io.BytesIO()
        np.savez_compressed(stream,**data)
        key=f'epoch-{epoch:02}/train-{shard:03}.npz'
        digest=hashlib.sha256(stream.getvalue()).hexdigest()
        assert digest==registry[key],key
        checks[key]=dict(sha256=digest,bytes=len(stream.getbuffer()))
    threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')}
    output.write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
                                      numba=numba.__version__,threads=threads,checks=checks),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--exact-later',action='store_true')
    args = parser.parse_args()
    if args.exact_later:exact_later(args.output)
    else:main(args.output)
