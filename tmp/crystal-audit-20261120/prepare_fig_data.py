from pathlib import Path
import json, csv
import numpy as np
root=Path('geometric-study/crystal-drift')
out=Path('tmp/crystal-audit-20261120')
fig=out/'figures'
def csvread(p):
    with p.open(newline='',encoding='utf8') as h:return list(csv.DictReader(h))
tim=csvread(root/'speed-results/timings-cpu.csv');prefixes=[]
for name,measure in [('drift','exact'),('flow1','flow1'),('flow16','flow16'),('retrieval','retrieval'),('uflow','uflow')]:
    for seed in ((2026,) if name=='uflow' else (11,23,37)):
        result=json.loads((root/'structure-matching'/f'multi-{measure}-{seed}.json').read_text())
        hit=np.isfinite(np.array(result['rms_by_candidate'],float))
        ms=[float(r['wall_ms']) for r in tim if r['cohort']=='multi' and r['batch']=='1' and r['seed']==str(seed) and r['method']==name]
        for k in range(1,6):prefixes.append(dict(method=name,seed=seed,k=k,targets=len(hit),hits=hit[:,:k].any(1).sum(),percent=hit[:,:k].any(1).mean()*100,median_ms=np.median(ms) if ms else ''))
with (fig/'binary_seed_prefixes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,prefixes[0]);w.writeheader();w.writerows(prefixes)
rows=json.loads((out/'full_chemistry.json').read_text())
with (fig/'native_recovery.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['cohort','method','n_targets','n_seeds','matcher_pct','volume_pct','max_site_pct'])
    for c in ('mp20','wbm'):
        for m in ('drift','flow1','flow16','retrieval','uflow'):
            ss=[v for v in rows if v['cohort']==c and v['method']==m]
            strict=json.loads((out/('strict-'+c+'.json')).read_text())
            fits=[r for r in strict if r['method']==m]
            w.writerow([c,m,ss[0]['n'],ss[0]['seeds'],ss[0]['estimate'],ss[1]['estimate'],100*np.mean([r['strict5']/r['cases'] for r in fits])])

