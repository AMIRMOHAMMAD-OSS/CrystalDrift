"""Independently check periodic distances and the two finite-ensemble energy conventions."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import expm


HERE=Path(__file__).resolve().parent
OUT=HERE/'comparison'


def physical(k, frac, species):
    a,b,c,d,e,f=np.asarray(k,dtype=np.float64)
    scale=np.sqrt(2.)
    g=expm(2*np.array([[a,d/scale,e/scale],[d/scale,b,f/scale],[e/scale,f/scale,c]]))
    z=species[species>0]
    x=np.stack((np.zeros(3),frac)) if frac.ndim==1 else frac[:len(z)]
    for radius in (1,2,4,8,16):
        shifts=np.array(list(product(range(-radius,radius+1),repeat=3)))
        best=np.full(3,np.inf)
        for i in range(len(z)):
            for j in range(i,len(z)):
                v=x[j].astype(float)-x[i]
                v=v-np.rint(v)+shifts
                ds=np.einsum('ni,ij,nj->n',v,g,v)
                if i==j:
                    ds[(v==0).all(1)]=np.inf
                category=int(z[i]!=z[0])+int(z[j]!=z[0])
                best[category]=min(best[category],ds.min())
        # Every unseen integer image has a fractional coordinate beyond radius+1/2.
        if np.linalg.eigvalsh(g).min()*(radius+.5)**2 > best.max():
            return np.r_[np.sqrt(np.linalg.det(g))/len(z),np.sqrt(best)]
    raise RuntimeError('Nearest-image search did not certify its bound')


def main():
    report={}
    for cohort,names in [('ab',('ab',20)),('multi',('multi',5))]:
        prefix='' if cohort=='ab' else 'multi-'
        with np.load(HERE/'data'/f'{prefix}test.npz') as ref:
            ids,z=ref['ids'],ref['z']
        run=HERE/'drift-results'/names[0]/'seed11.npz'
        vals=json.loads((OUT/'cache'/f'{cohort}-drift-11.json').read_text())['results']
        meta=json.loads((OUT/f'{cohort}-metadata.json').read_text())
        case_hash=meta['input_sha256']['test'][:12]
        fit_hash=meta['input_sha256']['train'][:12]
        ev_hash=meta['evaluator_sha256'][:12]
        test_feat=np.load(OUT/'cache'/f'{cohort}-test-{case_hash}-{ev_hash}.npy')
        train_feat=np.load(OUT/'cache'/f'{cohort}-train-{fit_hash}-{ev_hash}.npy')
        cols=[0,1,2] if cohort=='ab' else [0,1,2,3]
        std=np.log(train_feat[:,cols]).std(0)
        rng=np.random.default_rng(42071)
        picks=rng.choice(len(ids),size=8,replace=False)
        errors={'empirical_energy':[],'off_diagonal_spread':[],'fair_energy':[],'contact':[],'volume_log_error':[]}
        with np.load(run) as generated:
            k,fc=generated['k'],generated['f']
            for i in picks:
                x=np.stack([physical(k[i,j],fc[i,j],z[i]) for j in range(names[1])])
                y=test_feat[i]
                pred=np.log(np.maximum(x[:,cols],1e-8))/std
                target=np.log(y[cols])/std
                dist=np.linalg.norm(pred[:,None,:]-pred[None,:,:],axis=2)
                accuracy=np.linalg.norm(pred-target,axis=1).mean()
                n=len(x)
                spread=dist.sum()/(n*(n-1))
                energy=accuracy-dist.mean()/2
                stored=vals['conditional_energy_by_case'][i]
                stored_spread=vals['standardized_ensemble_spread_by_case'][i]
                checks={'empirical_energy':(energy,stored),'off_diagonal_spread':(spread,stored_spread),
                        'fair_energy':(accuracy-spread/2,stored-stored_spread/(2*n)),
                        'contact':((x[:,1:].min(1)<.5).mean(),vals['numeric_failure_or_short_contact_fraction_by_case'][i]),
                        'volume_log_error':(np.abs(np.log(x[:,0]/y[0])).mean(),vals['mean_abs_log_volume_ratio_by_case'][i])}
                for name,(v,w) in checks.items():
                    errors[name].append(abs(v-w))
                    np.testing.assert_allclose(v,w,atol=2e-6,rtol=1e-7)
        report[cohort]={'sample_seed':11,'random_case_ids':ids[picks].tolist(),
                        'draws_per_case':names[1],'maximum_absolute_differences':{n:float(max(v)) for n,v in errors.items()}}
    (OUT/'independent-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=1))


if __name__=='__main__':
    main()
