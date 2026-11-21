"""Rebuild the seed-averaged paired validation contrasts and retain each target's events."""
import csv,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(11,23,37)

def scores(fit,panel):
    if fit.name=='prior':
        path=fit/(panel+'-scores.npz')
    elif panel=='dev':
        chosen=fit/json.loads((fit/'selection.json').read_text())['chosen']
        path=chosen.with_name(chosen.stem+'-dev-scores.npz')
    else:
        path=fit/'frozen-confirm-scores.npz'
    with np.load(path) as data:
        accepted=np.isfinite(data['rms'])&(data['error']==0)
        native=accepted&(np.abs(data['log_volume_ratio'])<=np.log(1.1))
        return data['ids'],np.stack((accepted,native))

def main():
    rows=[];arrays={};fits=HERE/'fits'
    for panel in ('dev','confirm'):
        ids,prior=scores(fits/'prior',panel);arrays[panel+'_ids']=ids
        with np.load(fits/'prior'/(panel+'.npz')) as pool:
            sizes=(pool['Z'][:,0]>0).sum(1)
        arrays[panel+'_sites']=sizes
        for tier in ('small','medium'):
            arms={'prior':np.stack([prior]*len(SEEDS))}
            for arm in ('angular','contacts'):
                gathered=[]
                for seed in SEEDS:
                    these,flags=scores(fits/f'{tier}-{arm}-laplace-{seed}',panel)
                    if not np.array_equal(ids,these):raise ValueError('Target IDs differ')
                    gathered.append(flags)
                arms[arm]=np.stack(gathered)
            for name,baseline in [('angular','prior'),('contacts','prior'),('contacts','angular')]:
                for k in (1,5):
                    for ix,metric in enumerate(('scale','native')):
                        a=arms[name][:,ix,:,:k].any(-1);b=arms[baseline][:,ix,:,:k].any(-1)
                        d=a.astype(np.int8)-b.astype(np.int8)
                        arrays[f'{panel}_{tier}_{name}_{baseline}_{metric}_{k}']=d
                        for cohort,active in [('all',np.ones(len(ids),bool)),('N9plus',sizes>=9)]:
                            per_target=d[:,active].mean(0);r=np.random.default_rng(7331)
                            sample=r.integers(0,len(per_target),(5000,len(per_target)))
                            low,high=np.quantile(per_target[sample].mean(1),[.025,.975])
                            rows.append(dict(panel=panel,tier=tier,method=name,baseline=baseline,k=k,
                                metric=metric,cohort=cohort,targets=int(active.sum()),
                                delta_seed11_pp=100*d[0,active].mean(),delta_seed23_pp=100*d[1,active].mean(),
                                delta_seed37_pp=100*d[2,active].mean(),seed_mean_delta_pp=100*per_target.mean(),
                                low_pp=100*low,high_pp=100*high,mean_learned_rate=100*a[:,active].mean(),
                                mean_reference_rate=100*b[:,active].mean()))
    with (HERE/'validation-replications.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    np.savez_compressed(HERE/'validation-replications.npz',seeds=np.array(SEEDS),**arrays)
    print('Verified',len(rows),'three-seed paired validation contrasts.')

if __name__=='__main__':main()
