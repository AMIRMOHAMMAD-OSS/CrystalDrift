"""Directly rematch the native-volume anchoring sensitivity on the full development panel."""
import json,os
import numpy as np
from .select_weights import score
from .sampler import HERE

def main():
    here=HERE/'volume-dev';here.mkdir(exist_ok=True)
    path=HERE/'fits/small-contacts-laplace-11';s=json.loads((path/'selection.json').read_text())
    stem=s['chosen'].split('.')[0]
    with np.load(path/(stem+'-dev.npz')) as q,np.load(HERE/'fits/prior/dev.npz') as p:
        assert np.array_equal(q['ids'],p['ids'])
        L=q['L'];factor=np.cbrt(abs(np.linalg.det(p['L']))/abs(np.linalg.det(L)))
        out=here/'seed-volume.npz'
        np.savez_compressed(out,ids=q['ids'],F=q['F'],Z=q['Z'],L=L*factor[:,:,None,None])
    score(out)
    with np.load(here/'seed-volume-scores.npz') as a,np.load(path/(stem+'-dev-scores.npz')) as b:
        ah=np.isfinite(a['rms']);bh=np.isfinite(b['rms']);av=ah&(abs(a['log_volume_ratio'])<=np.log(1.1))
        bv=bh&(abs(b['log_volume_ratio'])<=np.log(1.1));d=av.any(1).astype(int)-bv.any(1)
        rng=np.random.default_rng(7331);ci=np.quantile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[.025,.975])
        rec=dict(checkpoint=s['sha256'],targets=len(d),draws=int(ah.size),changed_any_scale_flag=int(np.sum(ah!=bh)),
                 native_prior_volume=int(av.any(1).sum()),native_learned_volume=int(bv.any(1).sum()),
                 difference=float(d.mean()),paired_target_95=ci.tolist(),scaled_both=int(ah.any(1).sum()))
    (here/'counts.json').write_text(json.dumps(rec,indent=2))
if __name__=='__main__':main()

