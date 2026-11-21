"""Independent blind public-API check on 512 hash-selected targets across 12 arms."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from importlib.metadata import version

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data import unpack_gram
from standard_match import SETTINGS,make_structure


def pick(text):
    return hashlib.sha256(('crytaldrift-full-public-parity-v1/'+text).encode()).digest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    arg=argparse.ArgumentParser(description=__doc__);arg.add_argument('--compare-only',action='store_true')
    args=arg.parse_args();base=HERE/'mp20';shard=base/'blind-public-parity.npz'
    arms=[f'{m}-seed{s}' for m in ('drift','flow1','flow16','retrieval') for s in (11,23,37)]
    if not args.compare_only:
        try:
            import psutil
            proc=psutil.Process();proc.cpu_affinity([i for i in proc.cpu_affinity() if i not in (0,2,4)])
        except (ImportError,OSError,AttributeError):pass
        with np.load(base/'cohort.npz') as data:
            ref={k:data[k] for k in ('ids','z','k','f')}
        case=np.array(sorted(range(len(ref['ids'])),key=lambda i:pick(str(ref['ids'][i])))[:512])
        assigned=np.array([[int.from_bytes(pick(str(ref['ids'][i])+'/'+a)[:8],'big')%20 for a in arms] for i in case])
        pools=[tuple(np.load(base/a/(k+'.npy'),mmap_mode='r') for k in ('k','f')) for a in arms]
        matcher=StructureMatcher(**SETTINGS);num=len(case);begin=perf_counter()
        rms=np.full((num,len(arms)),np.nan);strict=np.zeros(rms.shape,bool)
        primitive_reduced=np.zeros(num,bool);ratio=np.full(rms.shape,np.nan)
        for row,i in enumerate(case):
            z=ref['z'][i];truth=make_structure(ref['k'][i],ref['f'][i],z)
            small=matcher._get_reduced_structure(truth,True,True)
            primitive_reduced[row]=(len(small)!=len(truth))
            for a,(ks,fs) in enumerate(pools):
                j=assigned[row,a];pred=make_structure(ks[i,j],fs[i,j],z)
                observed=matcher.get_rms_dist(pred,truth)
                if observed is not None:rms[row,a]=observed[0]
                strict[row,a]=matcher.fit(pred,truth)
                ratio[row,a]=np.log(pred.volume/truth.volume)
        tmp=shard.with_suffix('.writing')
        with tmp.open('wb') as out:
            np.savez_compressed(out,case=case,ids=ref['ids'][case],arms=np.array(arms),draw_index=assigned,
                public_rms=rms,public_strict=strict,raw_log_volume_ratio=ratio,
                primitive_target_site_count_changed=primitive_reduced,wall_seconds=perf_counter()-begin)
        tmp.replace(shard);print('Independent public calls complete',rms.size,flush=True)
    crosswalk=base/'complete-crosswalk-k20.npz'
    if not crosswalk.exists():return
    with np.load(shard) as d,np.load(crosswalk) as src:
        ids=d['case'];j=d['draw_index']; ix=(ids[:,None],np.arange(len(arms))[None,:],j)
        np.testing.assert_array_equal(d['ids'],src['ids'][ids]);np.testing.assert_array_equal(d['arms'],src['arms'])
        actual=src['rms'][ix];raw=d['public_rms'];strict=d['public_strict'];volume=d['raw_log_volume_ratio'];valid=np.isfinite(raw)
        np.testing.assert_array_equal(valid,np.isfinite(actual));np.testing.assert_allclose(raw[valid],actual[valid],rtol=1e-8,atol=1e-8)
        np.testing.assert_array_equal(src['strict_fit'][ix],strict)
        np.testing.assert_allclose(src['log_volume_ratio'][ix],volume,rtol=1e-6,atol=3e-7)
        report=dict(cases=len(ids),slots=raw.size,seed='crytaldrift-full-public-parity-v1',
            comparison='Independent public get_rms_dist and fit on each unfiltered matched target and prediction. Draw index assigned from ID and method/seed hash.',
            rms_hits=int(valid.sum()),strict_hits=int(strict.sum()),rms_hit_disagreements=0,maximum_site_disagreements=0,
            largest_normalized_rms_error=float(np.max(np.abs(raw[valid]-actual[valid]))),
            largest_log_volume_ratio_error=float(np.max(np.abs(src['log_volume_ratio'][ix]-volume))),
            reduced_target_site_count_changed=int(d['primitive_target_site_count_changed'].sum()),
            public_pymatgen=version('pymatgen'),crosswalk_sha256=sha(crosswalk),shard_sha256=sha(shard),
            elapsed_seconds=float(d['wall_seconds']))
    output=base/'blind-public-parity.json';output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
