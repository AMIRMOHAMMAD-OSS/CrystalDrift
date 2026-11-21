"""Reconstruct recovery at each wall-time budget from first-hit indices and original timing rows."""
import csv
import statistics
from collections import defaultdict

import numpy as np

from compare import write_json
from frontier import HERE, OUT, SEEDS


def rows(path):
    with path.open(encoding='utf8') as f:return list(csv.DictReader(f))


def main():
    timing=defaultdict(list)
    for file in (HERE/'speed-results').glob('timings-*.csv'):
        for r in rows(file):
            timing[r['cohort'],r['device'],r['method'],int(r['seed']),int(r['batch'])].append(float(r['wall_ms']))
    assert all(len(rep)==15 for rep in timing.values())
    first=defaultdict(list)
    for r in rows(OUT/'first-hit-by-case.csv'):
        cap=int(r['observation_cap'])
        # Failures take an index strictly beyond the available observation window.
        index=int(r['first_hit_candidate'] or cap+1)
        first[r['cohort'],r['criterion'],r['method'],int(r['seed'])].append((index,cap))
    first={k:np.asarray(v) for k,v in first.items()}
    max_err=0.;n_checks=0
    for r in rows(OUT/'fixed-time.csv'):
        cohort,criterion,method,device=[r[c] for c in ('cohort','criterion','method','device')]
        model='drift' if method in ('raw','exact') else method
        batch=int(r['batch']); budget=float(r['budget_ms'])
        per_seed=[]
        for seed in SEEDS:
            threshold,cap=first[cohort,criterion,method,seed].T
            cost=statistics.median(timing[cohort,device,model,seed,batch])/batch
            recoverable=(threshold<=cap) & (threshold*cost<=budget)
            per_seed.append(recoverable.mean())
        expected=statistics.mean(per_seed)
        max_err=max(max_err,abs(expected-float(r['recovery_fraction'])))
        n_checks+=1
    if max_err>1e-12:raise ArithmeticError(f'Fixed-time reconstruction differs by {max_err}')
    write_json(OUT/'independent-check.json',dict(fixed_time_rows_checked=n_checks,
        timing_cells_checked=len(timing),repetitions_per_cell=15,max_absolute_recovery_error=max_err,
        check='Directly test whether first observed hit index times independently computed statistics.median latency fits the budget. Enforce right censoring.'))
    print(n_checks,'fixed-time rows verified. Maximum error',max_err)


if __name__=='__main__':
    main()

