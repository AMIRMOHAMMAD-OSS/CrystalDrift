"""Prove lookup-cache ownership, and compare all 480 validation seeds before/after reuse."""
import gc,hashlib,json,weakref
import numpy as np
from .sampler import HERE,V3,ROOT,Sampler
from .train import validation_service,sha

def fingerprint(s,indices,z):
    h=hashlib.sha256()
    for i in indices:
        seed=int(np.random.SeedSequence([38471,i]).generate_state(1)[0])
        for v in s.seeds(z[i],5,seed):
            for x in (v.L,v.F,v.Z,v.orbit,v.J):h.update(x.tobytes())
            h.update(str((v.group,v.template,v.donor,v.cell_donor)).encode())
    return h.hexdigest()

def main():
    ids=json.loads((V3/'protocol/splits.json').read_text())['dev_ids']
    with np.load(ROOT/'scaled/mp20-data/val.npz') as f:
        z=f['z'];look={str(x):i for i,x in enumerate(f['ids'])}
    take=[look[x] for x in ids]
    s=Sampler();ref=weakref.ref(s);cached=fingerprint(s,take,z)
    owner=s.shortlist;s.shortlist=s._lookup_shortlist
    uncached=fingerprint(s,take,z)
    s.shortlist=owner;del owner;del s;gc.collect();released=ref() is None
    service=validation_service('cpu',sha(V3/'protocol/splits.json'))
    shortened=fingerprint(service,take,z);repeated=fingerprint(service,take,z)
    assert released and cached==uncached==shortened==repeated
    result=dict(seeds=480,full_index_released=released,inference_index_rows=len(service.rows),
        cached_sha256=cached,uncached_sha256=uncached,reused_sha256=shortened,repeat_sha256=repeated)
    (HERE/'lookup-cache-lifetime.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()

