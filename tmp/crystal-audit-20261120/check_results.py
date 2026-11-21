from pathlib import Path
import sys, csv, json, hashlib
from collections import Counter
import numpy as np
root=Path(sys.argv[1]).resolve()
out=Path(__file__).parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def perf(m): return np.isfinite(np.asarray(m['rms_by_candidate'],dtype=float))
def pct(v): return [round(float(x)*100,4) for x in np.atleast_1d(v)]
def dump(**kw): print(json.dumps(kw,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)))
def bootstrap(vals):
    samples=np.random.default_rng(9121).integers(len(vals),size=(10000,len(vals)))
    return pct([np.mean(vals),*np.quantile(vals[samples].mean(1),[.025,.975])])
# Analyse retained prediction matrices, including volume agreement for the same matched candidate.
figrows=[]; contrasts=[]; seeds=(11,23,37); paths=root/'scaled'/'runs'
for arm,src,tag in [('mp20','mp20-data','test'),('wbm','wbm-data','test'),('mp20-to-wbm','wbm-data','test'),('mp20-to-alex','alex-validation-transfer','val')]:
    with np.load(root/'scaled'/src/(tag+'.npz')) as a:
        ids=a['ids']; truth=dict(zip(ids, a['k'][:,:3].astype('float64').sum(1)))
    hits={}; vr={}; perseed={}
    for method in ['drift','flow1','flow16','retrieval','uflow']:
        if method=='uflow':
            if arm not in ('mp20','wbm'):continue
            files=list((root/'scaled'/'external'/arm).glob('uflow-*-matching.json'))
        else: files=sorted((paths/arm).glob(f'{method}-seed*-{tag}-matching.json'))
        rawlist=[]; gated=[]; donors=[]; variations=[]
        for f in files:
            m=read(f); hit=perf(m); ids0=m['material_ids']
            datapath=f.with_name(f.name.replace('-matching.json','.npz'))
            with np.load(datapath) as a:
                assert ids0==a['ids'].tolist()
                k=a['k'][:,:5,:3].astype('float64').sum(2)
                error=k-np.array([truth[x] for x in ids0])[:,None]
                gated.append((hit & (np.abs(error)<=np.log(1.1))).any(1))
                variations.append(np.median(k.std(axis=1,ddof=1)))
                if 'donor_ids' in a: donors.append([len(set(line)) for line in a['donor_ids']])
            rawlist.append(hit.any(1))
        hit=np.array(rawlist)
        hits[method]=hit;vr[method]=np.array(gated);perseed[method]=hit.sum(1).tolist()
        for metric,arr in [('matcher',hits[method]),('with_volume',vr[method])]:
            est,lo,hi=bootstrap(arr.mean(0))
            figrows.append(dict(cohort=arm,method=method,metric=metric,estimate=est,low=lo,high=hi,n=len(ids0),seeds=len(arr)))
        dump(cohort=arm,method=method, counts=perseed[method],volume_counts=np.sum(gated,axis=1),median_within_target_log_volume_sd=variations,one_distinct_donor_draws=(None if not donors else np.mean(np.array(donors)==1)))
    for m in ['flow1','flow16','retrieval']:
        both=hits['drift']&hits[m];bon=hits['drift']&~hits[m];other=~hits['drift']&hits[m]
        value=bootstrap(hits['drift'].mean(0)-hits[m].mean(0))
        contrasts.append(dict(cohort=arm,base=m,delta=value[0],low=value[1],high=value[2],drift_only=bon.sum(1).tolist(),other_only=other.sum(1).tolist(),both=both.sum(1).tolist()))
    dump(cohort=arm,deltas=contrasts[-3:])
for name,v in [('full_chemistry',figrows),('paired',contrasts)]:
    (out/(name+'.json')).write_text(json.dumps(v,indent=2))
# Raw binary timing observations and five drawn geometries; align IDs within each seed.
tim=rows(root/'speed-results'/'timings-cpu.csv')
bin_out=[]
for method in ['exact','flow1','flow16','retrieval','uflow']:
    matched=[];lat=[];admitted=[];timehits=[]; within_time=[]
    for seed in ((2026,) if method=='uflow' else seeds):
        hit=perf(read(root/'structure-matching'/f'multi-{method}-{seed}.json'))
        if method!='uflow':
            vals=np.array([float(r['wall_ms']) for r in tim if r['seed']==str(seed) and r['method']==({'exact':'drift'}.get(method,method)) and r['batch']=='1' and r['cohort']=='multi'])
            med=np.median(vals); n=min(5,int(5/med));lat.append(round(med,5));admitted.append(n);timehits.append(hit[:,:n].any(1));within_time.append(int((vals<=5).sum()))
        matched.append(hit)
    matched=np.array(matched);bs=[bootstrap(matched[:,:,:k].any(2).mean(0)) for k in (1,5)]
    fixed=bootstrap(np.mean(timehits,axis=0)) if timehits else None
    dump(binary=method,match1=bs[0],match5=bs[1],median=lat,budget=admitted,below5_observations=within_time,fixed=fixed)
    for metric,data in [('one',bs[0]),('five',bs[1]),('time',fixed)]:
        if data:
            est,lo,hi=data
            bin_out.append(dict(method=method,metric=metric,estimate=est,low=lo,high=hi))
(out/'binary_audit.json').write_text(json.dumps(bin_out,indent=2))
# Input cohort construction and independent physical summaries.
def compkeys(zz):
    from math import gcd
    return {tuple((k,v//g) for k,v in cs.items()) for row in zz for cs in [Counter(row[row>0])] for g in [int(np.gcd.reduce(list(cs.values()))) ]}
splits={}
for src in ('mp20-data','wbm-data'):
    formulas={};nrecs={}; hashes={}
    for s in ('train','val','test'):
        p=root/'scaled'/src/(s+'.npz')
        with np.load(p) as a:
            z=a['z'];formulas[s]=compkeys(z); nrecs[s]=len(z); hashes[s]=set(a['ids'].tolist())
            if s=='train': dump(data=src,ge3=sum(len(set(r[r>0]))>=3 for r in z))
    dump(data=src,sizes=nrecs,reduced_overlap={a+'-'+b:len(formulas[a]&formulas[b]) for a,b in [('train','test'),('train','val'),('val','test')]},ids_disjoint=not any(hashes[x]&hashes[y] for x,y in [('train','test'),('train','val'),('val','test')]))
    splits[src]=formulas
rr=rows(root/'de-novo/relaxation/results.csv'); phys=[r for r in rr if r['role']=='generated']
med=lambda k:np.median([float(r[k]) for r in phys])
short=[r for r in phys if float(r['final_min_covalent_radius_ratio'])<.6]
gap=np.array([float(r['delta_vs_reference_final_eV_per_atom']) for r in phys if r['delta_vs_reference_final_eV_per_atom']]);elem=[float(r['delta_vs_elements_eV_per_atom']) for r in rows(root/'de-novo/elemental-screen/candidate-gaps.csv')]
dump(relax=dict(stop=sum(r['converged']=='True' for r in phys),n=len(phys),nlowforce=sum(float(r['final_max_force_eV_per_A'])<=.1 for r in phys),short=len(short),short_converged=sum(r['converged']=='True' for r in short),abs_short=sum(float(r['final_min_contact_A'])<.5 for r in phys),med={k:med(k) for k in ['initial_max_force_eV_per_A','final_max_force_eV_per_A','initial_min_contact_A','final_min_contact_A','energy_change_eV_per_atom','relaxation_seconds']},ref=[len(gap),sum(gap>0),np.min(gap),np.median(gap),np.max(gap)],elem=[sum(np.array(elem)<0),np.median(elem)]))
with np.load(root/'de-novo/all-draws.npz') as a:
    z=a['z']; keys=[tuple(row[row>0]) for row in z]
    formulas=compkeys(z)
raw=rows(root/'de-novo/manifest.csv');nov=rows(root/'scaled/novelty/per-proposal.csv');assert [r['id'] for r in raw]==[r['generated_id'] for r in nov]
clean=[float(r['shortest_contact'])>=.7 for r in raw]
rep=np.array([r['new_representative']=='True' for r in nov]); nonmatch=np.array([not r['full_mp20'] and not r['wbm_native_20'] for r in nov])
dump(denovo=dict(draws=len(raw),distinct_reduced=len(formulas),distinct_exact=len(set(keys)),missing_reduced=sum(r['reduced_formula_in_training']=='False' for r in raw),abs_short=sum(float(r['shortest_contact'])<.5 for r in raw),short7=len(clean)-sum(clean),reps=rep.sum(),both_absent=nonmatch.sum(),joint=(rep&nonmatch&clean).sum()))

