"""Aggregate fully scored common cohorts; bootstrap target IDs with fitted seeds paired."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHODS=['drift','flow1','flow16','retrieval']
SEEDS=[11,23,37]
ARMS=[f'{m}-seed{s}' for m in METHODS for s in SEEDS]


def interval(value,values):
    lo,hi=np.nanquantile(values,[.025,.975])
    return dict(mean=float(value),lo=float(lo),hi=float(hi))


def by_case(rms,strict,volume):
    good=np.isfinite(rms)
    best=np.where(good,rms,np.inf).min(2)
    return {'rms_recovery':good.any(2),'strict_recovery':strict.any(2),
            'rms_and_volume_1.10':(good&(np.abs(volume)<=np.log(1.10))).any(2),
            'rms_total':np.where(np.isfinite(best),best,0.)}


def describe(rms,strict,volume,iterations):
    n=rms.shape[0];metrics=by_case(rms,strict,volume); names=list(metrics)
    raw=np.stack([metrics[key].reshape(n,len(METHODS),len(SEEDS)) for key in names],axis=3)
    seed_means={m:{str(s):{} for s in SEEDS} for m in METHODS}
    for a,(m,s) in enumerate((m,s) for m in METHODS for s in SEEDS):
        for key,v in metrics.items():
            if key!='rms_total':seed_means[m][str(s)][key]=float(v[:,a].mean())
        nhits=metrics['rms_recovery'][:,a].sum()
        seed_means[m][str(s)]['mean_best_rms_among_recovered']=float(metrics['rms_total'][:,a].sum()/nhits) if nhits else None
    mean=raw.mean(axis=(0,2));samples=np.empty((iterations,*mean.shape))
    seed_samples=np.empty((iterations,len(METHODS),len(SEEDS),len(names)))
    hit_input=metrics['rms_recovery'].astype(float)
    rms_input=metrics['rms_total']
    rng=np.random.default_rng(741123)
    for left in range(0,iterations,60):
        b=min(60,iterations-left);weight=rng.multinomial(n,np.full(n,1./n),size=b)
        per=(weight@raw.reshape(n,-1)).reshape(b,len(METHODS),len(SEEDS),len(names))/n
        # Within each seed the denominator is its own recovered-case count.
        # For a bootstrap draw that has zero hits for a seed the equal-seed
        # statistic is undefined and omitted from that statistic's interval.
        hits=per[:,:,:,0]
        per[:,:,:,-1]=np.divide(per[:,:,:,-1],hits,where=hits>0,out=np.full_like(hits,np.nan))
        seed_samples[left:left+b]=per;samples[left:left+b]=per.mean(2)
    mean[:,-1]=(rms_input.sum(0)/hit_input.sum(0)).reshape(len(METHODS),len(SEEDS)).mean(1)
    names[-1]='mean_best_rms_among_recovered'
    overall={m:{key:interval(mean[i,j],samples[:,i,j]) for j,key in enumerate(names)}
             for i,m in enumerate(METHODS)}
    contrasts={m:{key:interval(mean[0,j]-mean[i,j],samples[:,0,j]-samples[:,i,j])
                  for j,key in enumerate(names)} for i,m in enumerate(METHODS) if i}
    by_seed={m:{str(seed):{k:interval(seed_means[m][str(seed)][k],seed_samples[:,i,s,j])
                    for j,k in enumerate(names)} for s,seed in enumerate(SEEDS)} for i,m in enumerate(METHODS)}
    return dict(cases=n,methods=overall,per_seed=seed_means,per_seed_intervals=by_seed,paired_drift_minus=contrasts,
        conditional_rms_convention='Equal arithmetic mean of within-seed mean best normalized RMS among recovered targets.')


def read_mp(idx,last):
    n=len(idx);rms=np.empty((n,12,last),float);strict=np.empty(rms.shape,bool);vol=np.empty(rms.shape,np.float32)
    present=np.zeros((n,last),bool);audits=dict(unique_pairs=0,forwarded=0,public_api_comparisons=0,exceptions=0)
    for file in sorted((HERE/'mp20/tiles').glob('*.npz')):
        if not file.name.startswith('cases'):continue
        lo,hi=map(int,file.stem.split('-draws')[0].replace('cases','').split('-'))
        if lo>=n:continue
        with np.load(file) as d:
            a,b=int(d['first']),int(d['last']);b=min(b,last);end=min(n,hi)
            if a>=b:continue
            np.testing.assert_array_equal(d['ids'][:end-lo],idx[lo:end])
            np.testing.assert_array_equal(d['arms'],ARMS)
            if present[lo:end,a:b].any():raise RuntimeError(f'Duplicate result tile {file}')
            present[lo:end,a:b]=True
            for target,key in ((rms,'rms'),(strict,'strict_fit'),(vol,'log_volume_ratio')):
                target[lo:end,:,a:b]=d[key][:end-lo,:,:b-a]
            for key in ('unique_pairs','forwarded','public_api_comparisons'):audits[key]+=int(d[key])
            audits['exceptions']+=sum(json.loads(str(d['errors'])).values())
    if not present.all():return None
    return rms,strict,vol,audits


def write(path,obj):
    t=path.with_suffix('.writing');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=int,default=10000)
    args=p.parse_args();iters=args.bootstrap
    if (HERE/'wbm/test-crosswalk.npz').exists():
        with np.load(HERE/'wbm/test-crosswalk.npz') as w:
            np.testing.assert_array_equal(w['arms'],ARMS)
            out={str(k):describe(w['rms'][:,:,:k],w['strict_fit'][:,:,:k],w['log_volume_ratio'][:,:,:k],iters)
                 for k in (1,5)}
        write(HERE/'wbm/revision-summary.json',out)
    with np.load(HERE/'mp20/cohort.npz') as raw:idx=raw['ids']
    for n,tag in ((128,'prefix'),(len(idx),'full')):
        for b in (1,5,20):
            path=HERE/'mp20'/f'{tag}-k{b}.json'
            if path.exists() and 'per_seed_intervals' in json.loads(path.read_text()):continue
            done=read_mp(idx[:n],b)
            if done is None:continue
            rms,strict,volume,audit=done
            if n==128 and b<=5:
                for j,arm in enumerate(ARMS):
                    src=json.loads((ROOT/'scaled/runs/mp20'/(arm+'-test-matching.json')).read_text())
                    np.testing.assert_array_equal(idx[:128],src['material_ids'])
                    np.testing.assert_allclose(rms[:,j,:],np.asarray(src['rms_by_candidate'],float)[:,:b],equal_nan=True,atol=1e-8)
            stats=describe(rms,strict,volume,iters)
            stats.update(draws=b,source_split='official MP-20 test',cohort=tag,audit=audit,
                         case_id_sha256=hashlib.sha256('\n'.join(idx[:n]).encode()).hexdigest(),
                         bootstrap='10,000 paired ID resamples; recovery indicators seed-averaged per material; conditional RMS means formed within each seed then equally averaged')
            write(path,stats);print(tag,b,[round(stats['methods'][m]['rms_recovery']['mean'],4) for m in METHODS],flush=True)


if __name__=='__main__':main()
