"""Validate and compact a completed test crosswalk for archival."""
import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from report import HERE,ROOT,ARMS,read_mp,write


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--draws',type=int,choices=[1,5,20],default=20)
    p.add_argument('--audit',type=Path,help='Optional published independent strict-mp20.json for the original prefix')
    args=p.parse_args();base=HERE/'mp20'; source=ROOT/'scaled/mp20-data/test.npz'
    with np.load(base/'cohort.npz') as d:
        ids=d['ids'];z=d['z'];lk=d['k'][:,:3].sum(1,dtype=float);ix=d['release_index']
    with np.load(source) as raw:
        for field,prepared in (('ids',ids),('z',z)):np.testing.assert_array_equal(raw[field][ix],prepared)
        assert len(np.unique(ix))==len(raw['ids'])==len(ids)==9046
    result=read_mp(ids,args.draws)
    if result is None:raise RuntimeError('Await all target/draw tiles before writing a final crosswalk.')
    rms,strict,volume,audit=result
    engine_hash=hashlib.sha256((HERE/'match_full.py').read_bytes()+(HERE/'prescreen.py').read_bytes()).hexdigest()
    tiles=[]
    for tile in sorted((base/'tiles').glob('cases*-draws*.npz')):
        if int(tile.stem.split('-draws')[1].split('-')[0])>=args.draws:continue
        with np.load(tile) as raw:
            if str(raw['signature'])!=engine_hash or str(raw['pymatgen'])!='2024.4.13':
                raise ValueError(f'Unexpected implementation or package version in {tile.name}')
        tiles.append(tile.name)
    if audit['exceptions'] or (strict&~np.isfinite(rms)).any():raise ValueError('Inspect candidate exceptions or inconsistent hits')
    for j,arm in enumerate(ARMS):
        pool_info=json.loads((base/arm/'ready.json').read_text())
        for name,checksum in pool_info['outputs'].items():
            if sha(base/arm/name)!=checksum:raise ValueError(f'Candidate pool hash mismatch: {arm}/{name}')
        with np.load(ROOT/'scaled/runs/mp20'/(arm+'-test.npz')) as old:
            np.testing.assert_array_equal(old['ids'],ids[:128])
            k=np.load(base/arm/'k.npy',mmap_mode='r');f=np.load(base/arm/'f.npy',mmap_mode='r')
            np.testing.assert_array_equal(k[:128,:5],old['k']);np.testing.assert_array_equal(f[:128,:5],old['f'])
        v=k[:,:args.draws,:3].sum(2,dtype=float)-lk[:,None]
        np.testing.assert_allclose(v,volume[:,j],rtol=1e-6,atol=3e-7)
        legacy=json.loads((ROOT/'scaled/runs/mp20'/(arm+'-test-matching.json')).read_text())
        nr=min(5,args.draws)
        np.testing.assert_allclose(rms[:128,j,:nr],np.asarray(legacy['rms_by_candidate'],float)[:,:nr],equal_nan=True,atol=1e-8)
    strict_legacy=None
    if args.audit and args.draws>=5:
        for old in json.loads(args.audit.read_text()):
            arm=f"{old['method']}-seed{old['seed']}"
            if arm in ARMS:
                sub=strict[:128,ARMS.index(arm),:5]
                assert int(sub[:,0].sum())==old['strict1'] and int(sub.any(1).sum())==old['strict5']
        strict_legacy=sha(args.audit)
    file=base/f'complete-crosswalk-k{args.draws}.npz'
    with file.with_suffix('.writing').open('wb') as stream:
        np.savez_compressed(stream,ids=ids,z=z,arms=np.array(ARMS),rms=rms,
            strict_fit=strict,log_volume_ratio=volume,release_index=ix,draw_index=np.arange(args.draws))
    file.with_suffix('.writing').replace(file)
    note=dict(cohort='Complete 9,046-ID released MP-20 test split',test_sha256=sha(source),
        official_ids=9046,methods=4,seeds=[11,23,37],draws_per_target=args.draws,
        candidate_slots=rms.size,archived=file.name,archived_sha256=sha(file),published_128x5_prefix_preserved=True,
        outcome_audit=audit,tiles_checked=len(tiles),all_tile_matcher_hash=engine_hash,
        all_candidate_pool_hashes_verified=True,original_prefix_rms_verified=True,volume_log_formula_verified=True,
        independent_strict_128_audit_sha256=strict_legacy,
        packages={k:version(k) for k in ['pymatgen','numba','numpy','scipy']},python=platform.python_version(),
        methods_and_inputs={a:json.loads((base/a/'ready.json').read_text()) for a in ARMS},
        implementation={f:sha(HERE/f) for f in ['generate.py','match_full.py','prescreen.py','report.py','seal.py']},
        dependency_hashes={f:sha(ROOT/f) for f in ['scaled/family.py','scaled/experiment.py','scaled/evaluate.py',
                                                   'standard_match.py','data.py']})
    write(base/f'complete-crosswalk-k{args.draws}-manifest.json',note)
    print(file,'candidate slots',rms.size,'public API checks',audit['public_api_comparisons'],flush=True)


if __name__=='__main__':main()
