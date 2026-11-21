"""Seal immutable first-pass training shards, and check repeated writes did not alter the primary bank."""
import hashlib,json
from .sampler import HERE
from .prepare import CACHE

def main():
    paths=sorted(CACHE.glob('train-*.npz'))
    rows={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    for p in (CACHE/'calibration.npz',CACHE/'inventory.json'):
        rows[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    text=json.dumps(rows,indent=1,sort_keys=True)+'\n';out=HERE/'first-pass-sha256.json'
    if out.exists() and out.read_text()!=text:raise RuntimeError('The original bank changed')
    out.write_text(text)
if __name__=='__main__':main()

