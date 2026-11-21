"""Merge three complete WBM fits and score matched-candidate strict geometry sensitivities."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from match_full import ARMS,HERE,ROOT,MATCHER,make_structure,pin


def run(arm):
    pin()
    folder=ROOT/'scaled/runs/wbm' if arm.endswith('seed11') else HERE/'wbm'
    src=folder/(arm+'-test.npz')
    with np.load(src) as a:r={k:a[k] for k in ('ids','z','f','k')}
    audit=json.loads((folder/(arm+'-test-matching.json')).read_text())
    ids=r['ids'];np.testing.assert_array_equal(ids,audit['material_ids'])
    with np.load(ROOT/'scaled/wbm-data/test.npz') as d:
        lookup={s:i for i,s in enumerate(d['ids'])};ix=[lookup[s] for s in ids]
        rk,rf,rz=[d[k][ix] for k in ('k','f','z')]
    np.testing.assert_array_equal(rz,r['z'])
    rms=np.asarray(audit['rms_by_candidate'],float)
    fit=np.zeros(rms.shape,bool)
    dv=r['k'][:,:,:3].sum(2,dtype=float)-rk[:,:3].sum(1,dtype=float)[:,None]
    for i in range(len(ids)):
        truth=make_structure(rk[i],rf[i],rz[i]);memo={}
        for j in range(5):
            key=r['k'][i,j].tobytes(),r['f'][i,j].tobytes()
            if key not in memo:
                pred=make_structure(r['k'][i,j],r['f'][i,j],rz[i])
                memo[key]=bool(MATCHER.fit(pred,truth))
            fit[i,j]=memo[key]
    if (fit&~np.isfinite(rms)).any():raise ValueError('Unexpected max-site match without an RMS success')
    return ids,rms,fit,dv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    pin();folder=HERE/'wbm'
    # Small unchanged seed-11 score/matcher records make the common summary
    # interface usable while all original candidate tensors remain at source.
    import sys,shutil,subprocess
    for split in ('test','val'):
        for method in ('drift','flow1','flow16','retrieval'):
            stem=f'{method}-seed11-{split}'
            for suffix in ('.score.json','-matching.json'):
                dest=folder/(stem+suffix)
                orig=ROOT/'scaled/runs/wbm'/(stem+suffix)
                if dest.exists() and orig.read_bytes()!=dest.read_bytes():raise ValueError(str(dest))
                if not dest.exists():shutil.copyfile(orig,dest)
    sys.path.insert(0,str(ROOT/'scaled'))
    from match import summarize
    summarize(folder)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=pin) as ex:parts=list(ex.map(run,ARMS))
    ids=parts[0][0]
    for p in parts:np.testing.assert_array_equal(p[0],ids)
    out=folder/'test-crosswalk.npz'
    with out.with_suffix('.writing').open('wb') as stream:
        np.savez_compressed(stream,ids=ids,arms=np.array(ARMS),rms=np.stack([p[1] for p in parts],axis=1),
            strict_fit=np.stack([p[2] for p in parts],axis=1),log_volume_ratio=np.stack([p[3] for p in parts],axis=1))
    out.with_suffix('.writing').replace(out);print(out,flush=True)


if __name__=='__main__':main()
