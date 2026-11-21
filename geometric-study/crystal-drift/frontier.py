"""Empirical recovery at fixed candidate and measured generation-time budgets."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import compare
import data


HERE = Path(__file__).resolve().parent
OUT = HERE/'frontier-results'
SEEDS = compare.SEEDS
METHODS = compare.METHODS
MILLISECONDS = (0.025, 0.05, 0.1, 0.25, 0.5, 1., 2., 5., 10., 25., 50., 100., 250.)


def environment_hits(cohort, method, seed, test, train):
    cache=OUT/'cache'/f'environment-{cohort}-{method}-{seed}.npz'
    scored=compare.OUTPUT/'cache'/f'{cohort}-{method}-{seed}.json'
    signature=json.loads(scored.read_text(encoding='utf8'))['signature']
    signature=json.dumps(signature,sort_keys=True)
    if cache.exists():
        with np.load(cache) as f:
            if f['signature'].item()==signature:
                return f['ids'],f['hit']
    path=compare.prediction_path(cohort,method,seed,'test')
    if path is None:
        k,frac,_=data.retrieval_baseline(test,train,20,seed)
    else:
        with np.load(path) as pred:
            np.testing.assert_array_equal(pred['ids'],test['ids'])
            k,frac=pred['k'],pred['f']
    cap=20 if cohort=='ab' else 5
    k,frac=k[:,:cap],frac[:,:cap]
    n=len(k)
    values=data.geometry_features(k.reshape(-1,6),frac.reshape((n*cap,)+frac.shape[2:]),
                                  np.repeat(test['z'],cap,axis=0)).reshape(n,cap,4)
    truth=test['observed_features'][:,None,:]
    hit=(np.isfinite(values).all(-1) & (np.abs(np.log(np.maximum(values,1e-8)/truth)).max(-1)<=np.log(1.15)))
    expected=json.loads(scored.read_text(encoding='utf8'))['results']['has_draw_within_15pct_environment_fraction']
    np.testing.assert_allclose(hit.any(1).mean(),expected,rtol=0,atol=1e-12)
    with cache.with_suffix('.writing').open('wb') as target:
        np.savez_compressed(target,signature=signature,ids=test['ids'],hit=hit)
    cache.with_suffix('.writing').replace(cache)
    return test['ids'],hit


def matcher_hits(cohort,method,seed):
    f=HERE/'structure-matching'/f'{cohort}-{method}-{seed}.json'
    if not f.exists():return None
    row=json.loads(f.read_text(encoding='utf8'))
    ids=np.asarray(row['material_ids'])
    hit=np.isfinite(np.asarray(row['rms_by_candidate'],dtype=float))
    if hit.shape!=(len(ids),5):raise ValueError(f'Wrong match array shape at {f}')
    for k in (1,5):
        np.testing.assert_allclose(hit[:,:k].any(1).mean(),row['summary'][f'match_rate_at_{k}'])
    return ids,hit


def percentile_summary(case_values,bootstrap):
    n=len(case_values)
    mean=case_values.mean(0)
    lo,hi=np.quantile((bootstrap@case_values)/n,(.025,.975),axis=0)
    return mean,lo,hi


def make_prefix_table(input_map):
    rows,raw=[],[]
    for (cohort,metric,method),(ids,hits) in input_map.items():
        # Three seeds x structures x draws. Indicators concern the observed ordered prefix.
        by_prefix=np.logical_or.accumulate(hits,axis=2)
        if not np.all(np.diff(by_prefix.astype(np.int8),axis=2)>=0):raise AssertionError('Nonmonotone prefix')
        n=len(ids)
        boot=np.random.default_rng(741123).multinomial(n,np.full(n,1/n),size=10000).astype(float)
        p,low,high=percentile_summary(by_prefix.mean(0),boot)
        for j in range(hits.shape[2]):
            context={'cohort':cohort,'criterion':metric,'method':method,'candidates':j+1,'n_structures':n}
            rows.append({**context,'recovery_fraction':p[j],'low':low[j],'high':high[j]})
            for i,seed in enumerate(SEEDS):
                raw.append({**context,'seed':seed,'number_recovered':int(by_prefix[i,:,j].sum()),
                            'recovery_fraction':by_prefix[i,:,j].mean()})
    return rows,raw


def first_hit_rows(input_map):
    rows=[]
    for (cohort,metric,method),(ids,hits) in input_map.items():
        for j,seed in enumerate(SEEDS):
            hit=hits[j]
            first=np.where(hit.any(1),hit.argmax(1)+1,0)
            for mid,h in zip(ids,first):
                rows.append({'cohort':cohort,'criterion':metric,'method':method,'seed':seed,'id':mid,
                             'observation_cap':hit.shape[1],'first_hit_candidate':int(h) if h else ''})
    return rows


def collect(allow_pending):
    out={};pending=[]
    ev=compare.digest(HERE/'data.py')
    for cohort,methods in METHODS.items():
        test,_=compare.load_split(cohort,'test',ev)
        train,_=compare.load_split(cohort,'train',ev)
        condition_ids={}
        for method in methods:
            for metric in ('environment','structure'):
                items=[]
                for seed in SEEDS:
                    result=environment_hits(cohort,method,seed,test,train) if metric=='environment' else matcher_hits(cohort,method,seed)
                    if result is None:pending.append(f'{cohort}/{metric}/{method}/{seed}')
                    else:items.append(result)
                if len(items)!=len(SEEDS):continue
                if metric not in condition_ids:condition_ids[metric]=items[0][0]
                for ids,_ in items:np.testing.assert_array_equal(ids,condition_ids[metric])
                out[cohort,metric,method]=(condition_ids[metric],np.stack([h for _,h in items]))
    if pending and not allow_pending:raise RuntimeError('Missing matcher outputs: '+', '.join(pending))
    return out,pending


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available',action='store_true',help='Report completed three-seed arms while matching runs.')
    args=parser.parse_args()
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    collection,pending=collect(args.available)
    prefix,seeds=make_prefix_table(collection)
    compare.write_csv(OUT/'fixed-candidate.csv',prefix)
    compare.write_csv(OUT/'fixed-candidate-per-seed.csv',seeds)
    compare.write_csv(OUT/'first-hit-by-case.csv',first_hit_rows(collection))
    compare.write_json(OUT/'metadata.json',{'missing_matcher_outputs':pending,
        'match':'pymatgen StructureMatcher with five generated draws per model and target',
        'environment':'All four physical features are within a ratio [1/1.15,1.15] of the reference',
        'censoring':'Blank first_hit_candidate means no hit within the observed draw cap',
        'bootstrap':'Percentile interval from 10000 draws of structure IDs after averaging the three fitted seed indicators',
        'oracle':'Both stopping criteria require the held-out MP-20 reference geometry. Deployment would need an independently measured ranking or screening procedure.'})
    print('Scored',len(collection),'complete three-seed arms; pending matcher files',len(pending),flush=True)


if __name__=='__main__':
    main()
