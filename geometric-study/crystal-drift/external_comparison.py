"""Compare the released uFlowCSP checkpoint and local controls at five draws per target."""
import hashlib
import json
import numpy as np

import compare
import data
from frontier import matcher_hits


def run(cohort,require_match=False):
    out=compare.OUTPUT/f'external-{cohort}';out.mkdir(exist_ok=True)
    ev=compare.digest(compare.HERE/'data.py')
    full,tehash=compare.load_split(cohort,'test',ev)
    train,trhash=compare.load_split(cohort,'train',ev)
    ix=np.arange(len(full['ids'])) if cohort=='ab' else np.asarray(sorted(range(len(full['ids'])),
        key=lambda i:hashlib.sha256(full['ids'][i].encode()).hexdigest())[:128])
    test={k:v[ix] for k,v in full.items()}
    prefix='' if cohort=='ab' else 'multi-'
    evaluate=data.evaluate_predictions if cohort=='ab' else data.evaluate_multi
    n=len(test['ids']);boot=np.random.default_rng(741123).multinomial(n,np.full(n,1/n),size=10000).astype(float)
    scores={};seeds_table=[];case_table=[];aggregate=[];paired=[]
    for method in (*compare.METHODS[cohort],'uflow'):
        keys=(2026,) if method=='uflow' else compare.SEEDS
        metrics=[]
        for s in keys:
            file=compare.HERE/'uflowcsp-results'/f'{prefix}seed2026-euler1.npz' if method=='uflow' else compare.prediction_path(cohort,method,s,'test')
            signature=dict(source=compare.digest(file) if file else f'retrieval seed {s}, first 5 of 20',test=tehash,train=trhash,evaluator=ev)
            cache=out/f'cache-{method}-{s}.json'
            previous=json.loads(cache.read_text()) if cache.is_file() else {}
            if previous.get('signature')==signature:
                record=previous['scores']
            elif cohort=='multi' and method!='uflow':
                # These local case metrics were already evaluated at five draws on all 1,229 targets.
                existing=json.loads((compare.OUTPUT/'cache'/f'{cohort}-{method}-{s}.json').read_text())
                assert existing['signature']['reference']==[trhash,tehash] and existing['signature']['evaluator_sha256']==ev
                if file:assert existing['signature']['input']==signature['source']
                record={k:np.asarray(existing['results'][k])[ix].tolist() for k in compare.METRICS.values()}
                compare.write_json(cache,dict(signature=signature,scores=record))
            else:
                if file:
                    with np.load(file) as z:
                        np.testing.assert_array_equal(z['ids'],test['ids']);np.testing.assert_array_equal(z['z'],test['z'])
                        k,f=z['k'][:,:5],z['f'][:,:5]
                else:
                    k,f,_=data.retrieval_baseline(full,train,20,s);k,f=k[ix,:5],f[ix,:5]
                if k.shape!=(n,5,6):raise ValueError(f'Unexpected five-draw shape {k.shape}')
                record=evaluate(k,f,test,train)
                compare.write_json(cache,dict(signature=signature,scores=record))
            metric={name:np.array(record[k],float) for name,k in compare.METRICS.items()}
            if not np.all(np.isfinite(np.array(list(metric.values())))):raise ValueError(f'Nonfinite scores for {method}')
            mx=matcher_hits(cohort,method,s)
            if mx:
                ids,hit=mx;np.testing.assert_array_equal(ids,test['ids'])
                metric['matcher_at_1']=hit[:,0].astype(float);metric['matcher_at_5']=hit.any(1).astype(float)
            elif require_match:raise FileNotFoundError(f'Missing structural matches for {method} {s}')
            metrics.append(metric)
            seeds_table.append(dict(method=method,seed=s,n_targets=n,draws=5,
                **{name:float(v.mean()) for name,v in metric.items()}))
        for name in metrics[0]:
            across=np.mean([m[name] for m in metrics],axis=0)
            scores[method,name]=across
            aggregate.append(dict(method=method,metric=name,n_models=len(keys),draws=5,**compare.interval_row(across,boot)))
            for mid,v in zip(test['ids'],across):case_table.append(dict(method=method,metric=name,id=str(mid),value=v))
    for (method,name),each in scores.items():
        if method=='uflow' or ('uflow',name) not in scores:continue
        paired.append(dict(contrast='uflow minus '+method,metric=name,**compare.interval_row(scores['uflow',name]-each,boot)))
    compare.write_csv(out/'groups.csv',aggregate);compare.write_csv(out/'case-values.csv',case_table)
    compare.write_csv(out/'per-seed.csv',[{**{k:'' for row in seeds_table for k in row},**r} for r in seeds_table])
    compare.write_csv(out/'paired-uflow-minus.csv',paired)
    print('Five-draw',cohort,'energy:',{k:round(v.mean(),4) for (k,m),v in scores.items() if m=='energy'})
    compare.write_json(out/'metadata.json',dict(partition='official test',cases=n,draws=5,
        local_fits=list(compare.SEEDS),external_draw_seed=2026,external_checkpoint_fits=1,
        bootstrap='Average local fits per target, then resample paired reference IDs 10000 times.',
        external='Full-MP20 pretrained published uFlowCSP checkpoint; different training and inference resource budget.',
        standard_match_present=('uflow','matcher_at_5') in scores,
        official_replay_record=f'../../uflowcsp-results/{prefix}replay.json'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-match',action='store_true')
    p.add_argument('--cohort',choices=('ab','multi','both'),default='both')
    args=p.parse_args()
    for cohort in (('ab','multi') if args.cohort=='both' else (args.cohort,)):run(cohort,args.require_match)

