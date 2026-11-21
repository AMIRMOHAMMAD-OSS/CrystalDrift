"""Contrasts of changed capacity or training supervision at frozen validation weights."""
from pathlib import Path
import numpy as np
from .report_validation import features,compare,write
from .sampler import HERE
import json

def outcomes(fit,panel):
    chosen=fit/json.loads((fit/'selection.json').read_text())['chosen']
    path=chosen.with_name(chosen.stem+'-dev-scores.npz') if panel=='dev' else fit/'frozen-confirm-scores.npz'
    return features(path)

def main():
    fit=HERE/'fits';result=[]
    for panel in ('dev','confirm'):
        with np.load(fit/'prior'/(panel+'.npz')) as q:sizes=(q['Z'][:,0]>0).sum(1)
        for a,b in [('reference-contacts-laplace-11','medium-contacts-laplace-11'),
                    ('large-angular-laplace-11','medium-angular-laplace-11'),
                    ('small-angular-laplace-11-systemaux','small-angular-laplace-11'),
                    ('small-radial4-laplace-11','small-angular-laplace-11'),
                    ('small-angular-laplace-11','small-angular-laplace-11-published'),
                    ('small-angular-laplace-11','small-angular-laplace-11-local')]:
            left,right=outcomes(fit/a,panel),outcomes(fit/b,panel)
            for k in (1,5):
                for cutoff in (1,9):
                    metrics=compare(left,right,sizes>=cutoff,k)
                    metrics['reference_hits']=metrics.pop('prior_hits')
                    metrics['reference_native_hits']=metrics.pop('prior_native_hits')
                    result.append(dict(panel=panel,model=a,reference=b,k=k,minimum_sites=cutoff,**metrics))
    write(HERE/'capacity-paired.csv',result)
    print('Rebuilt',len(result),'paired capacity, architecture and supervision contrasts.')
if __name__=='__main__':main()
