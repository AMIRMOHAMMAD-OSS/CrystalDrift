"""Score five ordered candidate structures per held-out ID with pymatgen."""
import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from standard_match import SETTINGS,make_structure


PATTERN=re.compile(r'^(?P<method>[a-zA-Z0-9]+)-seed(?P<seed>\d+)-(?P<tag>.*?)(?P<split>val|test)$')
BOOT_SEED=741123


def sha(path):
    with path.open('rb') as f:
        d=hashlib.sha256()
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
        return d.hexdigest()


def write(path,obj):
    tmp=path.with_suffix('.writing')
    tmp.write_text(json.dumps(obj,indent=1,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)


@lru_cache(maxsize=4)
def references(path,count):
    with np.load(path,allow_pickle=False) as a:
        ids=a['ids'].astype(str)
        if len(set(ids))!=len(ids):raise ValueError(f'Duplicate reference ID in {path}')
        indices=sorted(range(len(ids)),key=lambda j:hashlib.sha256(ids[j].encode()).digest())[:count]
        output=tuple(a[k][indices] for k in ('ids','z','k','f'))
    return (*output,sha(path))


def interval(indicators,bootstrap):
    a=np.asarray(indicators,dtype=float)
    case_avg=a.mean(0) if a.ndim==2 else a
    lo,hi=np.quantile((bootstrap@case_avg)/len(case_avg),[.025,.975])
    return dict(value=float(case_avg.mean()),low=float(lo),high=float(hi))


def score(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    fname,reference=job
    attrs=PATTERN.fullmatch(fname.stem).groupdict();seed=int(attrs['seed'])
    out=fname.with_name(fname.stem+'-matching.json')
    try:
        with np.load(fname,allow_pickle=False) as a:
            ids,z,k,frac=(a[n] for n in ('ids','z','k','f'))
    except (zipfile.BadZipFile,EOFError):
        print('Incomplete input; re-run after export:',fname.name,flush=True);return
    refids,rz,rk,rf,refhash=references(reference,len(ids))
    np.testing.assert_array_equal(ids,refids);np.testing.assert_array_equal(z,rz)
    if (z.shape!=(len(ids),20) or k.ndim!=3 or k.shape[:1]+k.shape[2:]!=(len(ids),6) or
            k.shape[1]<5 or frac.shape!=(len(ids),k.shape[1],20,3)):
        raise ValueError(f'Unexpected dimensions: {fname}')
    signature=dict(prediction_sha256=sha(fname),reference_sha256=refhash,matcher_version=version('pymatgen'),
                   settings=SETTINGS,script_sha256=sha(Path(__file__)),crystal_decoder_sha256=sha(HERE.parent/'standard_match.py'),
                   geometry_sha256=sha(HERE.parent/'data.py'))
    if out.is_file() and json.loads(out.read_text())['signature']==signature:return
    t=perf_counter();matcher=StructureMatcher(**SETTINGS)
    rms_by_case=[];exceptions=Counter()
    for i in range(len(ids)):
        truth=make_structure(rk[i],rf[i],rz[i]);memo={};rms=[]
        for j in range(5):
            key=k[i,j].tobytes(),frac[i,j].tobytes()
            if key not in memo:
                try:
                    gen=make_structure(k[i,j],frac[i,j],z[i])
                    result=matcher.get_rms_dist(gen,truth)
                    memo[key]=float(result[0]) if result is not None and np.isfinite(result[0]) else None
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
                    memo[key]=None;exceptions[type(exc).__name__]+=1
            rms.append(memo[key])
        rms_by_case.append(rms)
    hit=np.isfinite(np.asarray(rms_by_case,float))
    boot=np.random.default_rng(BOOT_SEED).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=10000)
    summary=dict(method=attrs['method'],seed=seed,partition=attrs['tag']+attrs['split'],cases=len(ids),candidates=5,
                 n_elements_histogram=dict(sorted(Counter(str(len(set(s[s>0]))) for s in z).items())))
    for b in (1,5):
        observed=hit[:,:b].any(1)
        summary[f'hits_at_{b}']=int(observed.sum());summary[f'rate_at_{b}']=interval(observed,boot)
    summary.update(unique_candidate_exceptions=dict(exceptions),wall_seconds=perf_counter()-t)
    write(out,dict(signature=signature,summary=summary,material_ids=ids.tolist(),
                   actual_sites=(z>0).sum(1).tolist(),rms_by_candidate=rms_by_case))
    print(fname.parent.name,fname.stem,summary['hits_at_1'],summary['hits_at_5'],'of',len(ids),
          'exceptions',sum(exceptions.values()),'time',round(summary['wall_seconds'],1),flush=True)


def summarize(folder):
    grouped=defaultdict(dict)
    for path in sorted(folder.glob('*-matching.json')):
        r=json.loads(path.read_text());s=r['summary']
        grouped[s['partition'],s['method']][s['seed']]=r
    for partition in sorted({key[0] for key in grouped}):
        groups={k[1]:v for k,v in grouped.items() if k[0]==partition}
        ids=None;scores={}
        for method,fits in groups.items():
            for seed,v in fits.items():
                if ids is None:ids=v['material_ids']
                if ids!=v['material_ids']:raise ValueError('Targets differ across compared methods')
                hits=np.isfinite(np.asarray(v['rms_by_candidate'],float))
                for n in (1,5):scores[method,seed,n]=hits[:,:n].any(1).astype(float)
        boot=np.random.default_rng(BOOT_SEED).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=10000)
        summaries={};contrasts={}
        for m,fits in sorted(groups.items()):
            seeds=sorted(fits)
            summaries[m]=dict(seeds=seeds,**{f'at_{n}':interval([scores[m,s,n] for s in seeds],boot) for n in (1,5)})
            shared=sorted(set(fits)&set(groups.get('drift',{})))
            if m!='drift' and shared:
                contrasts['drift minus '+m]=dict(seeds=shared,**{f'at_{n}':interval(
                    [scores['drift',s,n]-scores[m,s,n] for s in shared],boot) for n in (1,5)})
        write(folder/f'structure-recovery-{partition}.json',dict(partition=partition,n_targets=len(ids),methods=summaries,
            paired=contrasts,bootstrap='10000 resamples of held-out IDs; average the common fitted-seed indicators per case before resampling',
            incomplete='Seed lists record exactly the available fitted checkpoints.'))
        print('Summary',folder.name,partition,{m:(x['seeds'],round(x['at_5']['value'],4)) for m,x in summaries.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Directory containing deposited name-seedN-{val,test}.npz predictions')
    parser.add_argument('--data',type=Path,help='Reference split directory; defaults from last -to- leg of run folder name.')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();folder=args.run
    data=args.data or HERE/(folder.name.split('-to-')[-1]+'-data')
    jobs=[]
    for p in sorted(folder.glob('*.npz')):
        parsed=PATTERN.fullmatch(p.stem)
        if parsed:jobs.append((p,data/(parsed.group('split')+'.npz')))
    if not jobs:print('No prediction files yet',folder);return
    if args.workers==1:
        for job in jobs:score(job)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:list(pool.map(score,jobs))
    summarize(folder)


if __name__=='__main__':
    main()

