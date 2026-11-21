"""Score saved crystal generators once, then compare them over paired test structures."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

import data


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'comparison'
SEEDS = (11, 23, 37)
METHODS = {'ab': ('drift', 'exact', 'raw', 'flow1', 'flow4', 'flow16', 'retrieval'),
           'multi': ('drift', 'exact', 'flow1', 'flow16', 'retrieval')}
METRICS = {'energy': 'conditional_energy_by_case',
           'short_contact_or_failure': 'numeric_failure_or_short_contact_fraction_by_case',
           'finite': 'evaluable_fraction_by_case',
           'environment_hit': 'has_draw_within_15pct_environment_fraction_by_case',
           'absolute_log_volume_error': 'mean_abs_log_volume_ratio_by_case',
           'ensemble_spread': 'standardized_ensemble_spread_by_case'}
REPORT_METRICS = tuple(METRICS) + ('fair_energy',)
BOOTSTRAP_SEED = 741123
REPLICATES = 10000


def write_json(path, obj):
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_path(cohort, method, seed, partition):
    if method == 'retrieval':
        return None
    if method.startswith('flow'):
        folder = 'baseline-results' if cohort == 'ab' else 'baseline-multi-results'
        return HERE / folder / f'seed{seed}-euler{method[4:]}.npz'
    folder = {'raw':'raw', 'exact':cohort+'-exact', 'drift':cohort}[method]
    parent = 'drift-results' if partition=='test' else 'drift-validation'
    return HERE / parent / folder / f'seed{seed}.npz'


def load_split(cohort, name, evaluator_digest):
    prefix = '' if cohort == 'ab' else 'multi-'
    path = HERE / 'data' / f'{prefix}{name}.npz'
    checksum = digest(path)
    with np.load(path) as f:
        split = {k: f[k] for k in ('k', 'f', 'z', 'ids')}
    folder = OUTPUT / 'cache'
    cache = folder / f'{cohort}-{name}-{checksum[:12]}-{evaluator_digest[:12]}.npy'
    if not cache.exists():
        features = data.geometry_features(split['k'], split['f'], split['z'])
        if not np.isfinite(features).all() or (features <= 0).any():
            raise ValueError('Reference contains nonpositive or unmeasurable physical features')
        with cache.with_suffix('.writing').open('wb') as f:
            np.save(f, features)
        cache.with_suffix('.writing').replace(cache)
    split['observed_features'] = np.load(cache)
    return split, checksum


def run_score(cohort, method, seed, train, test, checksums, evaluator_digest, partition):
    path = prediction_path(cohort, method, seed, partition)
    if path is not None and not path.is_file():
        return None
    prefix='' if partition=='test' else 'val-'
    cache = OUTPUT / 'cache' / f'{prefix}{cohort}-{method}-{seed}.json'
    sig = {'input': digest(path) if path else 'retrieval drawn with absolute seed '+str(seed),
           'reference': checksums, 'evaluator_sha256': evaluator_digest}
    if cache.exists():
        previous = json.loads(cache.read_text(encoding='utf-8'))
        if previous.get('signature') == sig:
            return previous
    start = perf_counter()
    if path is None:
        k, f, _ = data.retrieval_baseline(test, train, 20, seed)
    else:
        with np.load(path) as arr:
            np.testing.assert_array_equal(arr['ids'], test['ids'])
            np.testing.assert_array_equal(arr['z'], test['z'])
            k, f = arr['k'], arr['f']
        if k.shape != (len(test['ids']), 20, 6):
            raise ValueError(f'Unexpected candidate budget or cell output shape at {path}: {k.shape}')
    evaluate = data.evaluate_predictions if cohort == 'ab' else data.evaluate_multi
    results = evaluate(k, f, test, train)
    for key in METRICS.values():
        if len(results[key]) != len(test['ids']):
            raise ValueError(f'Wrong number of cases for {key}')
    record = {'signature': sig, 'cohort': cohort, 'method': method, 'seed': seed,
              'elapsed_seconds': perf_counter()-start, 'results': results}
    write_json(cache, record)
    print(cohort, method, seed, 'energy', results['conditional_energy'], 'seconds', round(record['elapsed_seconds'],2), flush=True)
    return record


def bootstrap_estimates(values, draws):
    """Same ID multiplicities for every method, with missing energies explicitly omitted."""
    ok = np.isfinite(values)
    if ok.all():
        means = (draws @ values) / len(values)
        return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()
    denominator = draws[:,ok].sum(1)
    means = np.full(len(draws), np.nan)
    np.divide(draws[:,ok] @ values[ok], denominator, out=means, where=denominator>0)
    return float(values[ok].mean()), *np.nanquantile(means, [0.025, 0.975]).tolist()


def interval_row(values, draws):
    if not np.isfinite(values).any():
        return dict(value=None, low=None, high=None, cases=0)
    mean, low, high = bootstrap_estimates(values, draws)
    return dict(value=mean, low=low, high=high, cases=int(np.isfinite(values).sum()))


def summarize(cohort, train, test, records, methods, contrast_reference):
    seeds, aggregates, differences = [], [], []
    ids = test['ids']
    in_train = set(map(tuple, train['z']))
    seen = np.array([tuple(z) in in_train for z in test['z']])
    subgroups = {'all': np.ones(len(ids),dtype=bool), 'seen_formula': seen, 'unseen_formula': ~seen}
    if cohort == 'multi':
        subgroups['repeated_species'] = np.sum(test['z'] > 0, axis=1) > 2
    per_method = {}
    for method in methods:
        per_seed = []
        for seed in SEEDS:
            r = records[method,seed]['results']
            values={name:np.asarray(r[key],dtype=np.float64) for name,key in METRICS.items()}
            # U-statistic estimate of population energy. The primary empirical-distribution
            # score includes self-pairs. This post-hoc estimate omits them on each case.
            valid_draws=np.rint(r['draws']*values['finite']).astype(int)
            correction=np.full(len(ids),np.nan)
            np.divide(values['ensemble_spread'],2*valid_draws,out=correction,where=valid_draws>=2)
            values['fair_energy']=values['energy']-correction
            columns=np.column_stack([values[key] for key in REPORT_METRICS])
            per_seed.append(columns)
            means = dict(zip(REPORT_METRICS, np.nanmean(columns,axis=0).tolist()))
            np.testing.assert_allclose(means['energy'], r['conditional_energy'], rtol=1e-10)
            np.testing.assert_allclose(means['short_contact_or_failure'], r['numeric_failure_or_short_contact_fraction'])
            seeds.append({'cohort': cohort, 'method': method, 'seed': seed, 'cases': len(ids),
                          'draws_per_case': r['draws'], **means, 'pooled_rbf_mmd_squared': r['mean_draw_pooled_rbf_mmd2_v']})
        # Require all three fits to have an estimate for a given target/metric.
        per_method[method] = np.mean(per_seed, axis=0)
    per_case = []
    for method, matrix in per_method.items():
        for i, row in enumerate(matrix):
            per_case.append({'cohort':cohort,'method':method,'id':str(ids[i]),'formula_in_train':bool(seen[i]),
                             'sites':int((test['z'][i]>0).sum()), **dict(zip(REPORT_METRICS,row.tolist()))})
    for index,(group,mask) in enumerate(subgroups.items()):
        n=int(mask.sum())
        rng = np.random.default_rng(BOOTSTRAP_SEED+index)
        draws = rng.multinomial(n, np.full(n,1/n), size=REPLICATES).astype(np.float64)
        for method, matrix in per_method.items():
            for j,name in enumerate(REPORT_METRICS):
                context = {'cohort':cohort,'group':group,'method':method,'metric':name}
                aggregates.append({**context, **interval_row(matrix[mask,j], draws)})
                if method != contrast_reference:
                    paired = per_method[contrast_reference][mask,j] - matrix[mask,j]
                    differences.append({**context,**interval_row(paired,draws)})
        del draws
    return seeds, aggregates, differences, per_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=tuple(METHODS)+('all',),default='all')
    parser.add_argument('--split',choices=('test','val','both'),default='both')
    parser.add_argument('--available',action='store_true',help='Score completed runs and summarize a cohort when complete.')
    args=parser.parse_args()
    (OUTPUT/'cache').mkdir(parents=True,exist_ok=True)
    source_hash=digest(HERE/'data.py')
    cohorts=METHODS if args.cohort=='all' else [args.cohort]
    partitions=('test','val') if args.split=='both' else (args.split,)
    for cohort,partition in ((c,p) for c in cohorts for p in partitions):
        methods=METHODS[cohort] if partition=='test' else ('exact','drift')
        reference='drift' if partition=='test' else 'exact'
        prefix='' if partition=='test' else 'val-'
        train, train_hash = load_split(cohort,'train',source_hash)
        test, test_hash = load_split(cohort,partition,source_hash)
        expected = {(m,s) for m in methods for s in SEEDS}
        records = {}
        for method in methods:
            for seed in SEEDS:
                rec=run_score(cohort,method,seed,train,test,[train_hash,test_hash],source_hash,partition)
                if rec is not None: records[method,seed]=rec
        if set(records)!=expected:
            missing=sorted(expected-set(records));print('Pending',cohort,partition,missing,flush=True)
            if not args.available:raise RuntimeError(f'Missing {len(missing)} candidate files for {cohort}')
            continue
        outputs=summarize(cohort,train,test,records,methods,reference)
        for stem,rows in zip(('per-seed','groups',f'paired-{reference}-minus','per-case'),outputs):
            write_csv(OUTPUT/f'{prefix}{cohort}-{stem}.csv', rows)
        write_json(OUTPUT/f'{prefix}{cohort}-metadata.json', {'seeds':SEEDS,'replicates':REPLICATES,'bootstrap_seed':BOOTSTRAP_SEED,
                  'case_unit':'held-out structure ID','partition':partition,'method_order':methods, 'python':platform.python_version(),
                  'exact_arm':'exploratory amendment following the original test comparison; validation comparison fixed before viewing validation scores',
                  'numpy':np.__version__,'scipy':scipy.__version__, 'evaluator_sha256':source_hash,
                  'seen_definition':'exact ordered elements with their counts in the supplied primitive cell',
                  'repeated_species_definition':'both species present and total sites > 2',
                  'bootstrap':'paired percentile intervals after arithmetic mean across three fits per structure; conditional on the fitted models',
                  'per_seed_uncertainty':'separate estimates in the per-seed table',
                  'retrieval_seed':'11,23,37 independent draws from training-reference tie sets; no model was fit',
                  'energy_missing_rule':'mean over evaluable draws per case; contrast requires all fitted seeds on both arms',
                  'fair_energy':'post-hoc sensitivity check: subtract the mean off-diagonal pair distance divided by 2d from the empirical energy, where d is the finite draw count',
                  'input_sha256':{'train':train_hash,partition:test_hash}})


if __name__=='__main__':
    main()
