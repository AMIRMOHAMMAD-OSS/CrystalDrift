"""Aggregate the fixed cohorts; resample held-out IDs, keeping training seeds paired."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);args=p.parse_args()
    groups={}
    for part in ['val','test']:
        group={}; baseline_ids=None
        for method in ['drift','flow1','flow16','retrieval']:
            records=[json.loads((args.run/f'{method}-seed{s}-{part}.score.json').read_text()) for s in args.seeds]
            for r in records:
                if baseline_ids is None:baseline_ids=r['target_ids']
                assert baseline_ids==r['target_ids']
            group[method]=records
        n=len(baseline_ids);rng=np.random.default_rng(574083);boot=rng.integers(n,size=(10000,n))
        def interval(x):
            return dict(mean=float(x.mean()),case_bootstrap_95_percentile=np.quantile(x[boot].mean(1),[.025,.975]).tolist())
        values={m:np.mean([r['per_target_energy'] for r in recs],axis=0) for m,recs in group.items()}
        stats={}
        for m,recs in group.items():
            stats[m]=dict(conditional_energy=interval(values[m]),
                seed_conditional_energy=[r['conditional_energy'] for r in recs],
                contact_passing=interval(np.mean([r['per_target_contact_fraction'] for r in recs],axis=0)),
                log_volume_mae=float(np.mean([r['mean_absolute_log_volume_error'] for r in recs])),
                min_contact_mae_angstrom=float(np.mean([r['mean_absolute_min_contact_error_angstrom'] for r in recs])))
        groups[part]=dict(targets=n,stats=stats,drift_energy_minus={m:interval(values['drift']-x) for m,x in values.items() if m!='drift'})
    result={'seed_order':args.seeds,'bootstrap':'10000 case-resampled means; paired across arms, averaged over the listed training seeds before resampling',**groups}
    (args.run/'geometry-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for m,d in result['test']['stats'].items():print(m,d['conditional_energy'],d['contact_passing']['mean'])


if __name__=='__main__':main()
