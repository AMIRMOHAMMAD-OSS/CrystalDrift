"""Pair CrytalDrift and uFlowCSP's released raw draws by official MP-20 target ID."""
import csv
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
V2 = OUT.parent
ROOT = V2.parents[1]
SEEDS = (11, 23, 37)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def composition(row, reduced=False):
    elements, counts = np.unique(row[row > 0], return_counts=True)
    if reduced:
        counts = counts // reduce(gcd, counts)
    return tuple(zip(elements, counts))


def main():
    cohort = read_npz(ROOT / 'revision/full/mp20/cohort.npz')
    train = read_npz(ROOT / 'scaled/mp20-data/train.npz')
    uflow = read_npz(ROOT / 'revision/full/external/mp20/complete-crosswalk-k20.npz')
    angular = [read_npz(V2 / f'csp/test/angular-seed{seed}/outcomes.npz') for seed in SEEDS]
    for pool in [uflow, *angular]:
        np.testing.assert_array_equal(cohort['ids'], pool['ids'])
    np.testing.assert_array_equal(cohort['z'], uflow['z'])

    training_counts = [Counter(composition(row, reduced) for row in train['z']) for reduced in (False, True)]
    seen, reduced_seen = [np.array([composition(row, r) in cnt for row in cohort['z']])
                          for r, cnt in zip((False, True), training_counts)]
    n = (cohort['z'] > 0).sum(axis=1)
    elements = np.array([len(composition(row)) for row in cohort['z']])
    groups = {
        'all': np.ones(len(n), bool),
        'seen_exact': seen, 'unseen_exact': ~seen, 'seen_reduced': reduced_seen, 'unseen_reduced': ~reduced_seen,
        'sites1-4': n <= 4, 'sites5-8': (n >= 5) & (n <= 8),
        'sites9-12': (n >= 9) & (n <= 12), 'sites13-20': n >= 13,
        'elements1': elements == 1, 'elements2': elements == 2,
        'elements3': elements == 3, 'elements4+': elements >= 4,
    }
    a_matches = np.isfinite(np.stack([pool['rms'] for pool in angular], axis=1))
    u_matches = np.isfinite(uflow['rms'])
    with (OUT / 'paired_strata.csv').open('w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['stratum','targets','draws','mean_angular_pct','uflow_pct','uflow_minus_angular_pp',
                    'angular_seed','both','uflow_only','angular_only','neither'])
        for name, mask in groups.items():
            for k in (1, 5, 20):
                au = a_matches[mask, :, :k].any(axis=-1)
                uf = u_matches[mask, :k].any(axis=-1)
                summary = [name, len(uf), k, round(100 * au.mean(), 2), round(100 * uf.mean(), 2),
                           round(100 * (uf.mean() - au.mean()), 2)]
                for seed, arm in zip(SEEDS, au.T):
                    counts = [(uf & arm).sum(), (uf & ~arm).sum(), (~uf & arm).sum(), (~uf & ~arm).sum()]
                    w.writerow([*summary, seed, *counts])
    with (OUT / 'paired_common.csv').open('w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['draws','angular_seed','common_targets','angular_best_normalized_rms','uflow_best_normalized_rms'])
        for k in (1, 5, 20):
            u_rms = uflow['rms'][:, :k]
            u_best = np.where(np.isfinite(u_rms), u_rms, np.inf).min(1)
            for seed, pool in zip(SEEDS, angular):
                a_rms = pool['rms'][:, :k]
                a_best = np.where(np.isfinite(a_rms), a_rms, np.inf).min(1)
                common = np.isfinite(a_best) & np.isfinite(u_best)
                w.writerow([k, seed, common.sum(), a_best[common].mean(), u_best[common].mean()])

    with (OUT / 'paired_native.csv').open('w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['draws','model','match','native_volume_and_match','max_site_fit','mae_log_volume_all_draws'])
        for k in (1, 5, 20):
            for name, pool in zip(('angular11','angular23','angular37','uflow'), [*angular, uflow]):
                hit = np.isfinite(pool['rms'][:, :k])
                dr = pool['log_volume_ratio'][:, :k]
                w.writerow([k, name, hit.any(1).mean(), (hit & (np.abs(dr) <= np.log(1.1))).any(1).mean(),
                            pool['strict_fit'][:, :k].any(1).mean(), np.abs(dr).mean()])


if __name__ == '__main__':
    main()
