"""Independently reduce frozen target, clock, group and de novo records in a clean extraction."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import numpy as np

HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]

def load_json(p):return json.loads(p.read_text(encoding='utf8'))
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as fp:return list(csv.DictReader(fp))
def close(actual,expected):
    if expected is None:assert actual is None
    else:np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=3e-10)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda:fp.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()
def manifest():
    index=load_json(HERE/'EVIDENCE-MANIFEST.json')
    for name,r in index['files'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],name
    return dict(files=len(index['files']),scope=index['scope'],hashes_verified=True)

def submission():
    report=load_json(HERE/'CrytalDrift-build.json')
    pdf=HERE/report['pdf']
    archive=HERE/report['source']
    assert digest(pdf)==report['pdf_sha'] and digest(archive)==report['source_sha']
    assert [p['pass_number'] for p in report['passes']]==[1,2]
    assert not report['unresolved'] and not report['passes'][-1]['undefined']
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(report['members']) and z.testzip() is None
        for name in z.namelist():
            assert z.read(name)==(V3/'paper'/name).read_bytes(),name
    files=('author_report.py','author_brief.py','MAIN-COMPARISON.csv','SEED-RESULTS.csv','MORNING.md','MORNING.pdf',
           'CLAIM-AUDIT.md','CrytalDrift-claims.json',
           'evidence-index.json','MILESTONE-INITIAL.md','CHANGES.md')
    assert all((HERE/name).is_file() for name in files)
    claims=load_json(HERE/'CrytalDrift-claims.json')
    assert claims['report']==pdf.name and claims['sha256']==report['pdf_sha']
    assert claims['pages']==len(report['layout']) and claims['clean_source_members']==len(report['members'])
    evidence=load_json(HERE/'evidence-index.json')
    for name,expected in evidence.items():
        source=(V3/name).resolve()
        assert source.is_relative_to(V3) and digest(source)==expected,name
    restoration=HERE/'retained-run-settings.json'
    settings={}
    if restoration.is_file():
        for name,proof in load_json(restoration).items():
            path=(ROOT/name).resolve()
            assert path.is_relative_to(V3/'evaluation/raw') and digest(path)==proof['sha256'],name
            settings[proof['binding']]=settings.get(proof['binding'],0)+1
    replication=V3/'geometry/audits/replication-table-verification.json'
    original=digest(replication)
    subprocess.run([sys.executable,'-B','-m','lattice_symmetry.v3.geometry.check_replication_tables'],cwd=ROOT,check=True)
    assert digest(replication)==original,'Replication table receipt changed during clean replay'
    return dict(pdf_sha256=report['pdf_sha'],source_sha256=report['source_sha'],members=len(report['members']),
                pages=len(report['layout']),latex_passes=2,unresolved=report['unresolved'],overfull=report['overfull'],
                author_evidence_sources=len(evidence),restored_settings=settings,replication_table=load_json(replication))

def alignment(live=False):
    index=None if live else load_json(HERE/'EVIDENCE-MANIFEST.json')['files'];total=draws=0;arms={}
    pools=list((V3/'evaluation/scores').glob('*/*/outcomes.npz'))+list((V3/'uflow/scores').glob('*/*/*/outcomes.npz'))
    for outcome in sorted(pools):
        dest=outcome.parent
        if dest.is_relative_to(V3/'uflow/scores'):
            relative=dest.relative_to(V3/'uflow/scores');raw=V3/'uflow/predictions'/relative
        else:
            relative=dest.relative_to(V3/'evaluation/scores');raw=V3/'evaluation/raw'/relative
            if live:
                pointer=V3/'evaluation/raw-locations'/relative.parent/(relative.name+'.json')
                if pointer.exists():raw=Path(load_json(pointer)['actual'])
        sections=sorted(dest.glob('?????-?????.npz'))
        if not sections:continue
        with np.load(outcome) as f:whole={k:f[k] for k in ('ids','rms','strict_fit','log_volume_ratio','error')}
        offset=0
        for p in sections:
            input_path=raw/p.name;member=str(input_path) if live else input_path.relative_to(ROOT).as_posix()
            with np.load(p) as segment,np.load(input_path) as candidates:
                assert str(segment['source_sha'])==(digest(input_path) if live else index[member]['sha256']),member
                np.testing.assert_array_equal(candidates['ids'],segment['ids']);n=len(segment['ids'])
                for key,all_data in whole.items():np.testing.assert_array_equal(all_data[offset:offset+n],segment[key])
                total+=1;offset+=n;draws+=segment['rms'].size
        assert offset==len(whole['ids']);arms[dest.relative_to(V3).as_posix()]=list(whole['rms'].shape)
    return dict(shards=total,scored_draws=draws,arms=arms)

def checkpoints():
    out={}
    inventory=V3/'models/deployment-weights.json'
    if inventory.is_file():
        for spec in load_json(inventory)['files']:
            p=ROOT/spec['path'].replace('\\','/');assert p.stat().st_size==spec['bytes'] and digest(p)==spec['sha256']
    for base in (V3/'models/fits',V3/'uflow/fits'):
        for selection in sorted(base.glob('*/selection.json')):
            s=load_json(selection)
            if 'checkpoint_sha' in s:
                path=selection.parent/s['checkpoint'];expected=s['checkpoint_sha']
                winner=min(s['results'],key=lambda name:(-s['results'][name],name));assert path.stem==winner
            elif 'chosen' in s and 'sha256' in s:
                path=selection.parent/s['chosen'];expected=s['sha256']
            else:continue
            if not path.is_file():
                if str(expected).lower() in ('identity','none') or not expected:continue
                raise FileNotFoundError(path)
            if path.suffix!='.pt':continue
            assert path.resolve().is_relative_to(V3) and digest(path)==expected,path
            out[selection.parent.relative_to(V3).as_posix()]=expected
    for freeze in (V3/'evaluation').glob('*frozen.json'):
        for arm in load_json(freeze).get('methods',[]):
            if not arm.get('checkpoint') or not arm.get('sha'):continue
            p=(V3/arm['checkpoint']).resolve();assert p.is_relative_to(V3) and digest(p)==arm['sha'],arm['tag']
    manifest_files=load_json(HERE/'EVIDENCE-MANIFEST.json')['files']
    bank_hashes=load_json(V3/'models/fresh-bank-sha256.json')['shards']
    for relative,expected in bank_hashes.items():
        assert manifest_files['lattice_symmetry/v3/models/cache/'+relative]['sha256']==expected,relative
    for relative,spec in load_json(V3/'models/first-pass-sha256.json').items():
        assert manifest_files['lattice_symmetry/v3/models/cache/'+relative]==spec,relative
    for name,fit in load_json(V3/'models/terminal-fit-integrity.json').items():
        if 'optimizer' not in fit:continue
        optimizer=fit['optimizer']
        key=f'lattice_symmetry/v3/models/fits/{name}/resume.pt'
        assert manifest_files[key]=={'bytes':optimizer['bytes'],'sha256':optimizer['sha256']},name
        assert fit['full_generator_equal_to_terminal_optimizer'] and fit['all_adam_steps']==[optimizer['step']]
        assert fit['reconstructed_final_rng_equal'] and fit['distinct_anchor_positions']==fit['eligible_anchors']
        out[key]=optimizer['sha256']
    return out

def common_gaussian_inputs():
    base=V3/'uflow'
    receipt=load_json(base/'verified-noise-identity.json')
    totals={}
    for panel,expected in receipt['panels'].items():
        verified=0
        attempts=0
        originals=base/'predictions/prior-11/frozen'/panel
        for record_path in sorted((base/'streams'/panel).glob('*.json')):
            record=load_json(record_path)
            prior=originals/(record_path.stem+'.npz')
            assert digest(prior)==record['prior_shard_sha256']
            assert record['nonbitwise_cells']==record['nonbitwise_sites_draws']==0
            with np.load(prior) as f:
                ids=f['ids']
                species=f['Z']
            for arm in ('small-23','small-37','medium-23','medium-37'):
                candidate=base/'predictions'/arm/'frozen'/panel/prior.name
                with np.load(candidate) as f:
                    np.testing.assert_array_equal(f['ids'],ids)
                    np.testing.assert_array_equal(f['Z'],species)
                    assert str(f['noise_sha256'])==record['gaussian_sha256']
                verified+=1
                attempts+=len(ids)*record['draws_per_target']
        assert verified==expected['replicate_hashes_checked'] and attempts==4*expected['draws']
        totals[panel]=dict(verified_hashes=verified,identical_inputs=attempts)
    return totals

def bootstrap(delta,labels):
    rng=np.random.default_rng(7331)
    if labels is None:
        values,n=np.unique(delta,return_counts=True)
        replicates=rng.multinomial(len(delta),n/n.sum(),5000).dot(values)/len(delta)
    else:
        names,code=np.unique(labels,return_inverse=True)
        ids=rng.integers(len(names),size=(5000,len(names)))
        replicates=np.bincount(code,weights=delta)[ids].sum(1)/np.bincount(code)[ids].sum(1)
    return (np.percentile(replicates,[2.5,97.5])*100).tolist()

def historical_native():
    original=ROOT/'revision/full/external/mp20'
    ledger=load_json(original/'complete-manifest.json')
    for member,expected in ledger['tile_sha256'].items():assert digest(original/member)==expected,member
    assert digest(original/'complete-crosswalk-k20.npz')==ledger['crosswalk_sha256']
    base=V3/'uflow';prefix=load_json(base/'published-prefix-replay.json');gpu=load_json(base/'published-gpu-replay.json')
    assert digest(base/'published-prefix.npz')==prefix['prior_output_sha256']==gpu['prefix_sha256']
    all_probes=prefix['probes']+list(gpu['probes'].values())
    for branch,count in [('prefix',640),('remainder',180280)]:
        assert load_json(base/f'published-{branch}-identity.json')['separately_recoded_latents_verified']==count
    for row in all_probes:assert row['k_max']<1e-4 and row['fractional_max']<1e-4
    spec=load_json(base/'published-paired.json')
    with np.load(original/'complete-crosswalk-k20.npz') as gold:ids=gold['ids'].astype(str);species=gold['z']
    m,mv,ms=attempts(original/'complete-crosswalk-k20.npz',ids)
    p,pv,ps=attempts(base/'scores/published-prior/frozen/test/outcomes.npz',ids)
    assert m.shape==p.shape==(ledger['targets'],ledger['draws'])==(9046,20)
    positions={mid:i for i,mid in enumerate(ids)}
    scored=np.zeros_like(ms);generated=np.zeros_like(ms);min_volume=np.inf
    for member in ledger['tile_sha256']:
        with np.load(original/member) as tile:
            indices=np.array([positions[str(mid)] for mid in tile['ids']])
            if member.startswith('matches/'):
                sl=np.ix_(indices,np.arange(int(tile['first']),int(tile['last'])))
                assert not scored[sl].any();scored[sl]=True
                for arr,key in ((m,'rms'),(mv,'log_volume_ratio'),(ms,'strict_fit')):np.testing.assert_array_equal(arr[sl],tile[key])
            else:
                j=int(tile['draw']);assert not generated[indices,j].any()
                generated[indices,j]=True
                np.testing.assert_array_equal(species[indices],tile['z'])
                min_volume=min(min_volume,float(np.exp(tile['k'][:,:3].sum(1)).min()))
    assert scored.all() and generated.all();close(min_volume,spec['archived_learned_minimum_volume'])
    for k in (1,5,20):
        for name,a,b in [('scale',np.isfinite(m),np.isfinite(p)),('max_site',ms,ps),
                         ('native',np.isfinite(m)&(abs(mv)<=np.log(1.1)),np.isfinite(p)&(abs(pv)<=np.log(1.1)))]:
            h=a[:,:k].any(1);z=b[:,:k].any(1);expect=spec['metrics'][str(k)][name]
            assert (h.sum(),z.sum(),(h&~z).sum(),(z&~h).sum())==tuple(expect[v] for v in ('released_hits','zero_hits','gained','lost'))
            d=h.astype(float)-z;close(d.mean()*100,expect['delta_pp'])
            np.testing.assert_allclose(bootstrap(d,None),expect['ci95_pp'],atol=5.1e-6,rtol=0)
    return dict(historical_geometry_and_score_tiles=len(ledger['tile_sha256']),ordered_candidates=m.size,
                branch_counts=[gpu['cpu_prefix_draws'],gpu['cuda_decode_draws']],replayed_probe_draws=sum(r['cases'] for r in all_probes),
                metrics=spec['metrics'])

def attempts(path,order,with_errors=False):
    with np.load(path,allow_pickle=False) as f:
        ids=f['ids'].astype(str);assert len(ids)==len(set(ids));look=dict(zip(ids,range(len(ids))))
        idx=[look[s] for s in order];r=f['rms'][idx];v=f['log_volume_ratio'][idx];s=f['strict_fit'][idx]
        errors=f['error'][idx]!=0 if 'error' in f else np.zeros_like(r,dtype=bool)
        assert not (np.isfinite(r)&errors).any()
        assert r.shape==v.shape==s.shape and len(r)==len(order)
        return (r,v,s,errors) if with_errors else (r,v,s)

def target_strata(panel,ids,registry):
    files=ROOT/'scaled/mp20-data'
    if not (files/'train.npz').exists():return {}
    split='test' if panel=='test' else 'train' if panel in ('family','shared') else 'val'
    with np.load(files/(split+'.npz')) as d:
        index=dict(zip(d['ids'].astype(str),range(len(d['ids']))))
        atoms=d['z'][[index[mid] for mid in ids]]
    with np.load(files/'train.npz') as d:
        rows=d['z'][np.isin(d['ids'],registry['train_ids'])]
    composition=lambda z: tuple(sorted(int(a) for a in z if a>0))
    known={composition(z) for z in rows}
    sizes=(atoms>0).sum(1)
    seen=np.array([composition(z) in known for z in atoms])
    out={'large':sizes>=9,'seen':seen,'unseen':~seen}
    for lo,hi in ((1,4),(5,8),(9,12),(13,20)):
        out[f'{lo}-{hi}']=(sizes>=lo)&(sizes<=hi)
    duplicates=V3/'evaluation/duplicates.csv'
    if panel=='test' and duplicates.exists():
        overlap={r['id'] for r in read_csv(duplicates) if r['split']=='test' and r['earlier_training_match']}
        out['no_tight_training_match']=~np.isin(ids,list(overlap))
    return {name:mask for name,mask in out.items() if mask.any()}

def csp(report,registry):
    panel=report['panel'];ids=report['ids'];assert len(ids)==len(set(ids))
    clusters=None
    if panel in ('family','shared'):clusters=[registry['family_by_challenge_id'][s] for s in ids]
    elif panel.startswith('arrangement'):
        fam=load_json(V3/'geometry/audits/arrangement-frozen.json')['key_by_id'];clusters=[str(fam[s]) for s in ids]
    processed={};records={};comparisons={};subset_rows=0
    strata=target_strata(panel,ids,registry)
    for tag,paths in report['source'].items():
        channels=[attempts(ROOT/p.replace('\\','/'),ids,with_errors=True) for p in paths]
        for k in (1,5,20):
            if any(len(r[0])<k for r,_,_,_ in channels):continue
            hits=[];native=[];strict=[];rms=[];best=[]
            for r,v,s,errors in channels:
                vals=np.where(np.isfinite(r[:,:k]),r[:,:k],np.inf).min(1)
                matched=np.isfinite(vals);hits.append(matched);best.append(vals)
                native.append((np.isfinite(r[:,:k])&(abs(v[:,:k])<=np.log(1.1))).any(1));strict.append(s[:,:k].any(1))
                rms.append(float(vals[matched].mean()) if matched.any() else None)
            h=np.array(hits);n=np.array(native);sm=np.array(strict)
            key=f'all/{k}/{tag}';expected=report['quality'][key]
            arr=[v for v in rms if v is not None];avg=float(np.mean(arr)) if arr else None
            for actual,target in ((h.mean(),expected['mr']),(n.mean(),expected['native']),(sm.mean(),expected['strict']),
                                  (h.sum(1),expected['hits_per_seed']),(n.sum(1),expected['native_hits']),(avg,expected['rms_matched'])):close(actual,target)
            for aa,bb in zip(rms,expected['rms_by_seed']):close(aa,bb)
            scale_values=np.stack([v[:,:k] for _,v,_,_ in channels])
            failures=np.stack([e[:,:k] for _,_,_,e in channels])
            finite=scale_values[np.isfinite(scale_values)]
            median_volume=float(np.median(finite)) if len(finite) else None
            close(median_volume,expected['median_log_v']);close(failures.sum(),expected['errors'])
            processed[(k,tag)]=(h,n)
            records[key]=dict(targets=len(ids),hits=h.sum(1).tolist(),native=n.sum(1).tolist(),strict=sm.sum(1).tolist(),rms=rms,
                              errors=int(failures.sum()),median_log_volume_ratio=median_volume)
            for label,mask in strata.items():
                subset=report['quality'][f'{label}/{k}/{tag}']
                assert mask.sum()==subset['n']
                close(h[:,mask].sum(1),subset['hits_per_seed'])
                close(n[:,mask].sum(1),subset['native_hits'])
                close(sm[:,mask].mean(),subset['strict'])
                vols=scale_values[:,mask]
                finite=vols[np.isfinite(vols)]
                close(float(np.median(finite)) if len(finite) else None,subset['median_log_v'])
                close(failures[:,mask].sum(),subset['errors'])
                for disp,expected_rms in zip(best,subset['rms_by_seed']):
                    finite=mask&np.isfinite(disp)
                    close(float(disp[finite].mean()) if finite.any() else None,expected_rms)
                subset_rows+=1
    for key,ref in report['paired'].items():
        strat,sk,desc=key.split('/',2)
        if strat!='all':continue
        left,right=desc.split('--');a=processed[int(sk),left];b=processed[int(sk),right]
        for j,val in enumerate(('pp','native_pp')):
            delta=(a[j].astype(float)-b[j]).mean(0)
            close(100*delta.mean(),ref[val]);expected=ref['ci95_pp' if j==0 else 'native_ci95_pp']
            ci=bootstrap(delta,clusters);close(ci,expected)
            if j==0:
                close((a[0]&~b[0]).sum(1),ref['first_only']);close((b[0]&~a[0]).sum(1),ref['second_only'])
                comparisons[key]=[float(100*delta.mean()),*ci]
    return dict(targets=len(ids),recovery=records,paired_scale_pp=comparisons,
                subgroup_rows_verified=subset_rows,strata={name:int(mask.sum()) for name,mask in strata.items()})

def geometries():
    out={}
    for p in sorted((V3/'geometry/audits').glob('reachable-*.npz')):
        summary=load_json(p.with_suffix('.json'));counts={}
        with np.load(p) as d:
            assert len(d['ids'])==summary['ids'] and int(d['truth_available'].sum())==summary['truth_available']
            for k in (1,5,20):
                found={key:int(d[key][:,:k].any(1).sum()) for key in ('canonical','equivalent','family','group','fixed')}
                assert found==summary['reach_by_k'][str(k)];counts[str(k)]=found
        out['shortlist/'+summary['panel']]=counts
    for p in sorted((V3/'geometry/audits').glob('matched-group-*.json')):
        if 'tolerances' in p.stem:continue
        panel=p.stem.removeprefix('matched-group-')
        for tag,arm in load_json(p).items():
            source=V3/f'evaluation/scores/{panel}/{tag}/outcomes.npz'
            if not source.is_file():continue
            rows=arm['records']
            with np.load(source) as data:
                rms=data['rms'][:,:5];ms=data['ids'].astype(str);lookup=dict(zip(ms,range(len(ms))))
            hit=np.isfinite(rms).any(1);assert set(ms[hit])=={row['id'] for row in rows}
            for row in rows:
                i=lookup[row['id']];j=int(np.nanargmin(rms[i]));assert row['best_draw_zero_based']==j;close(rms[i,j],row['rms'])
            same=sum(row['reference']>0 and row['detected']==row['reference'] for row in rows)
            contained=sum(row['contains'] for row in rows)
            assert same==arm['recovered_with_correct_detected_group'] and len(rows)==arm['matched'] and contained==arm['contains_requested']
            out[panel+'/'+tag]=dict(N=len(ms),matched=len(rows),same_detected=same,contains_requested=contained)
    for summary in sorted((V3/'geometry/audits').glob('accepted-groups-*/summary.json')):
        panel=summary.parent.name.removeprefix('accepted-groups-')
        with np.load(summary.parent/'prior.npz') as f:baseline=f['detected'];base_ids=f['ids']
        for method,spec in load_json(summary).items():
            with np.load(summary.parent/(method+'.npz')) as d:
                ids=d['ids'];rms=d['rms'];group=d['detected'];reference=d['reference']
            np.testing.assert_array_equal(base_ids,ids);np.testing.assert_array_equal(group>=0,np.isfinite(rms))
            original=V3/f'evaluation/scores/{panel}/{method}/outcomes.npz'
            if original.exists():
                with np.load(original) as f:np.testing.assert_array_equal(f['ids'],ids);np.testing.assert_array_equal(f['rms'],rms)
            for k in (1,5,20):
                g=group[:,:k];same=(g==reference[:,None])&(g>0);hit=(g>=0).any(1)
                best=np.where(g>=0,rms[:,:k],np.inf).argmin(1)
                counts=dict(targets=len(ids),matched_targets=int(hit.sum()),matched_draws=int((g>=0).sum()),
                            failed_detections=int((g==0).sum()),correct_group_draws=int(same.sum()),
                            correct_group_any_draw=int(same.any(1).sum()),
                            correct_group_best_draw=int(same[np.arange(len(g)),best].sum()))
                if method!='prior':
                    new=hit&~(baseline[:,:k]>=0).any(1)
                    counts.update(added_vs_prior=int(new.sum()),correct_group_on_added=int((new&same.any(1)).sum()))
                assert counts==spec['rows'][str(k)]
                out[f'accepted/{panel}/{method}/{k}']=counts
    return out

def clocks():
    folder=V3/'evaluation/timing';out={};elapsed={}
    frozen=V3/'evaluation/clock-frozen.json'
    calls={r['tag']:int(bool(r.get('checkpoint') or r.get('u_tier'))) for r in load_json(frozen)['methods']} if frozen.is_file() else {}
    for p in sorted(folder.glob('cpu-*.npz')):
        with np.load(p) as d:
            if 'first_ms' not in d:continue
            ms=d['first_ms'];assert (ms>0).all() and ms.shape==(len(d['ids']),len(d['methods'])) and int(d['threads'])==1
            out[p.stem]={str(tag):dict(median_ms=float(np.median(ms[:,i])),p95_ms=float(np.percentile(ms[:,i],95)),targets=len(ms)) for i,tag in enumerate(d['methods'])}
    for p in sorted(folder.glob('*-10k.npz')):
        with np.load(p) as d:
            assert len(d['ids'])==10000 and (d['batch_seconds']>=0).all() and d['total_seconds']>=sum(d['batch_seconds'])-.001
            tag=p.stem.removesuffix('-10k');forwards=int(d['calls'])
            if tag in calls:assert forwards==len(d['batch_seconds'])*calls[tag]
            out[p.stem]=dict(n=10000,seconds=float(d['total_seconds']),batch=int(d['batch']),atoms=int(d['atoms']),measured_forwards=forwards)
    if (folder/'elapsed.csv').exists():
        flags={};draws={}
        for record in read_csv(folder/'elapsed.csv'):
            tag=record['method'];ms=int(record['ms'])
            with np.load(folder/(tag+'.npz')) as proposals,np.load(folder/(tag+'-scores.npz')) as match:
                np.testing.assert_array_equal(proposals['ids'],match['ids']);t=proposals['elapsed_ms'];r=match['rms']
                if tag in calls:assert int(proposals['calls'])==t.size*calls[tag]
            assert np.all(np.diff(t)>=0);finished=t<=ms;hit=(np.isfinite(r)&finished).any(1)
            assert int(hit.sum())==int(record['hits']) and int(finished.sum())==int(record['completed_draws']);close(hit.mean(),float(record['mr']))
            elapsed[tag+'/'+str(ms)]=[int(hit.sum()),int(finished.sum())]
            flags[tag,ms]=hit;draws[tag]=(t,r)
        for row in read_csv(folder/'draws.csv'):
            k=int(row['k']);r=draws[row['method']][1];assert int(np.isfinite(r[:,:k]).any(1).sum())==int(row['hits'])
        for row in read_csv(folder/'paired.csv'):
            k=int(row['ms']);delta=flags[row['first'],k].astype(float)-flags[row['second'],k]
            close(100*delta.mean(),float(row['pp']));close(bootstrap(delta,None),[float(row['low']),float(row['high'])])
        for row in read_csv(folder/'latency.csv'):
            t=draws[row['method']][0];inc=np.diff(t,axis=1,prepend=0)
            for k,value in [('first_median',np.median(t[:,0])),('first_p95',np.percentile(t[:,0],95)),
                           ('serial_median',np.median(inc)),('serial_p95',np.percentile(inc,95))]:close(value,float(row[k]))
    out['retrospective_elapsed']=elapsed;return out

def concat(paths,k):
    rows=[]
    for p in sorted(paths):
        with np.load(p,allow_pickle=False) as f:rows.append(f[k])
    return np.concatenate(rows)

def denovo():
    new=V3/'denovo';ref=ROOT/'expansion/denovo/scores';out={}
    for p in sorted((new/'results').glob('*.json')):
        spec=load_json(p)
        if 'arm' not in spec:continue
        arm=spec['arm'];rows=list((new/'scores/raw'/arm).glob('*.npz'));raw=lambda k:concat(rows,k)
        with np.load(new/f'results/{arm}-cases.npz') as d:
            cases={k:d[k] for k in d.files}
        valid=raw('comp_valid')&raw('struct_valid');np.testing.assert_array_equal(valid,cases['valid'])
        rows2=read_csv(new/f'novelty/{arm}.csv');assert list(cases['ids'])==[x['id'] for x in rows2]
        uniq=np.array([r['representative_index']==str(i) and not r['error'] for i,r in enumerate(rows2)])
        novel=np.array([not (r['matched_train_id'] or r['error']) for r in rows2]);np.testing.assert_array_equal(cases['unique'],uniq);np.testing.assert_array_equal(cases['novel'],novel)
        metrics=dict(n=len(valid),struct_valid=int(sum(raw('struct_valid'))),comp_valid=int(sum(raw('comp_valid'))),joint_valid=int(valid.sum()),
            unique=int(uniq.sum()),novel=int(novel.sum()),valid_unique_novel=int((valid&uniq&novel).sum()),
            coverage_precision=float(cases['precision'].sum()/len(valid)),coverage_recall=float(cases['recall'].mean()))
        ref_raw=list((ref/'raw/reference').glob('*.npz'))
        if ref_raw:
            from scipy.stats import wasserstein_distance
            for key,col in [('density','density'),('elements','nelem')]:
                metrics['w1_'+key]=float(wasserstein_distance(raw(col)[cases['property_indices']],concat(ref_raw,col)))
        fp=list((new/'scores/fp'/arm).glob('*.npz'));rf=list((ref/'fp/reference').glob('*.npz'))
        if fp and rf:
            from scipy.spatial import cKDTree
            tree=ast.parse((ROOT/'revision/denovo/diffcsp_constants.txt').read_text());const={x.targets[0].id:np.asarray(ast.literal_eval(x.value)) for x in tree.body if isinstance(x,ast.Assign)}
            def feats(pool):
                ok=concat(pool,'error')=='';comp=concat(pool,'comp')[ok];s=concat(pool,'struc')[ok]
                return np.nan_to_num((comp-const['CompScalerMeans'])/const['CompScalerStds']),s
            c,s=feats(fp);c0,s0=feats(rf)
            for a,b,original in [((c,s),(c0,s0),cases['precision']),((c0,s0),(c,s),cases['recall'])]:
                close_comp=cKDTree(b[0]).query(a[0],distance_upper_bound=np.nextafter(10.,np.inf),workers=2)[0]<=10
                close_struct=cKDTree(b[1]).query(a[1],distance_upper_bound=np.nextafter(.4,np.inf),workers=2)[0]<=.4
                np.testing.assert_array_equal(close_comp&close_struct,original)
        for k,v in metrics.items():close(v,spec[k])
        out[arm]=metrics
    groups=new/'catalog/spacegroup-summary.json'
    if groups.is_file():
        cat=groups.parent;obj=load_json(groups);out['groups']={}
        def dist(a,b):
            a=np.bincount(a,minlength=231).astype(float);a/=a.sum()
            b=np.bincount(b,minlength=231).astype(float);b/=b.sum();m=.5*(a+b)
            return float(np.sqrt(.5*sum((x[x>0]*np.log2(x[x>0]/m[x>0])).sum() for x in (a,b))))
        with np.load(cat/'reference-test.npz') as d:test=d['detected']
        with np.load(cat/'reference-chemistry-source.npz') as d:source=d['detected']
        for arm,spec in obj['arms'].items():
            with np.load(cat/(arm+'-spacegroups.npz')) as d:g=d['detected']
            n=int(np.sum(g>0));p1=int(np.sum(g==1));q=int(len(set(g[g>0])))
            assert (n,p1,q)==(spec['detected'],spec['p1'],spec['distinct_positive_types'])
            refdist=dist(g,test);srcdist=dist(g,source);equal=int(np.sum((g>0)&(g==source)))
            close(refdist,spec['js_distance_base2_to_test']);close(srcdist,spec['js_distance_base2_to_composition_sources']);assert equal==spec['equal_source_type']
            out['groups'][arm]=dict(n=len(g),detected=n,p1=p1,distinct=q,jensen_shannon_to_test=refdist,jensen_shannon_to_source=srcdist,equal_source=equal)
    return out

def relaxation():
    folder=V3/'denovo/screen'
    if not (folder/'by-arm.csv').is_file():return {}
    rows=read_csv(folder/'per-candidate.csv');result={}
    for s in read_csv(folder/'by-arm.csv'):
        if s['group']!='all':continue
        data=[x for x in rows if x['arm']==s['arm']]
        keys={'converged':'converged','complete_hulls':'complete_hull','thermally_screened':'energetic_pass',
              'unique':'unique','novel':'novel','sun_lower':'screened_sun','sun_upper':'possible_sun'}
        actual={key:sum(row[col]=='True' for row in data) for key,col in keys.items()}
        actual.update(n=len(data),evaluated=sum(bool(q['final_energy']) for q in data),
                      short=sum(bool(q['final_min_ratio']) and float(q['final_min_ratio'])<.6 for q in data),
                      candidate_core_seconds=sum(float(q['wall_seconds']) for q in data))
        for key,value in actual.items():close(value,float(s[key]))
        result[s['arm']]=actual
    groups=V3/'denovo/catalog/post-relaxation-groups.json'
    if groups.is_file():
        result['symmetry_retention']={}
        for arm,s in load_json(groups).items():
            records=s['records'];assert len(records)==s['n']
            for key,counts in s['counts'].items():assert Counter(str(row[key]) for row in records)==counts
            keep={label:sum(row['raw_'+label]>0 and row['raw_'+label]==row['relaxed_'+label] for row in records) for label in ('tight','loose')}
            assert keep==s['exact_type_retained'];result['symmetry_retention'][arm]=keep
    library=folder/'reference-library.json'
    if library.is_file():
        s=load_json(library);recs=[folder/'records'/(mid+'.json') for mid in s['phase_ids']]
        if all(p.is_file() for p in recs):
            values=[load_json(p) for p in recs];assert len(values)==s['phases'] and sum(v['converged'] for v in values)==s['converged']
            seconds=sum(v['wall_seconds'] for v in values);close(seconds,s['total_reference_core_seconds'])
            result['competing_phase_library']=dict(name=s['name'],count=len(values),converged=s['converged'],seconds=seconds)
    return result

def ordered_cifs(external=None):
    catalog=V3/'denovo/catalog';out={}
    for record in sorted(catalog.glob('*.json')):
        info=load_json(record)
        if 'archive_sha256' not in info:continue
        arm=info['arm'];path=ROOT/info['archive']
        if external is not None:path=external/info['archive'].replace('\\','/').split('/')[-1]
        pool=V3/f'denovo/pools/{arm}.npz'
        assert digest(path)==info['archive_sha256'] and digest(pool)==info['source_sha256']
        with np.load(pool) as p,np.load(catalog/(arm+'-spacegroups.npz')) as g,ZipFile(path) as arc:
            rows=list(csv.DictReader(io.StringIO(arc.read('attempts.csv').decode('utf8'))))
            ids=[row['id'] for row in rows]
            assert len(set(ids))==len(ids)==info['attempts']==10000
            np.testing.assert_array_equal(ids,p['ids']);np.testing.assert_array_equal(ids,g['ids'])
            np.testing.assert_array_equal([int(row['order']) for row in rows],np.arange(1,len(ids)+1))
            for col,key in [('requested_group_if_any','requested'),('detected_group_at_0p01_A','detected'),('detected_hall','hall')]:
                np.testing.assert_array_equal([int(row[col]) for row in rows],g[key])
            for col,key in [('composition_source','chemistry_source'),('pattern_donor','donor'),('cell_donor','cell_donor')]:
                expected=p[key].astype(str) if key in p else np.full(len(ids),'')
                np.testing.assert_array_equal([row[col] for row in rows],expected)
            files=[r['cif'] for r in rows if r['cif']]
            assert all(r['cif']==('cifs/'+r['id']+'.cif') if r['cif'] else bool(r['export_error']) for r in rows)
            assert len(files)==info['cifs'] and set(arc.namelist())==set(files)|{'README.txt','attempts.csv'}
            assert len(arc.namelist())==len(files)+2 and arc.testzip() is None
            chosen=sorted((r['id'] for r in rows if r['cif']),key=lambda mid:hashlib.sha256(('v3-cif-read/'+mid).encode()).digest())[:32]
            assert chosen==[r['id'] for r in info['read_back']]
            assert all(r['metric_relative']<2e-12 and r['fractional_max']<2e-8 for r in info['read_back'])
        out[arm]=dict(attempts=len(rows),files=len(files),readback_records=len(chosen),nested_crc=True)
    assert len(out)==7
    return out

def main(a):
    if a.archive is not None:
        if a.extract_new is None:raise ValueError('An archive replay needs a fresh extraction directory')
        folder=a.extract_new.resolve();folder.mkdir(parents=True,exist_ok=False)
        with ZipFile(a.archive) as archive:
            names=archive.namelist();assert len(names)==len(set(names))
            index_name='lattice_symmetry/v3/release/EVIDENCE-MANIFEST.json';index=json.loads(archive.read(index_name))
            assert set(names)==set(index['files'])|{index_name}
            for name in names:
                if not (folder/name).resolve().is_relative_to(folder) or Path(name).is_absolute():raise ValueError(name)
            written=0;next_guard=0
            for name in names:
                output=folder/name
                output.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source,output.open('xb') as dest:
                    for chunk in iter(lambda:source.read(4<<20),b''):
                        if a.host_reserve_mib and written>=next_guard and Path('/mnt/c').is_dir():
                            v=os.statvfs('/mnt/c');free=v.f_bavail*v.f_frsize
                            if free<a.host_reserve_mib*(1<<20):raise OSError(f'Windows volume has {free/(1<<20):.1f} MiB free during clean extraction')
                            next_guard=written+(32<<20)
                        dest.write(chunk);written+=len(chunk)
        print('Extracted into fresh directory',folder,flush=True)
        child=[sys.executable,str(folder/'lattice_symmetry/v3/release/replay.py'),'--verify-manifest']
        if a.output:child+=['--output',str(a.output.resolve())]
        if a.panels:child+=['--panels',*a.panels]
        subprocess.run(child,cwd=folder,check=True);return
    import scipy
    receipt={'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__}}
    if a.live_bindings:
        result=alignment(live=True);assert a.output is not None
        a.output.write_text(json.dumps(result,indent=1)+'\n');print(result['shards'],result['scored_draws']);return
    if a.verify_manifest:
        receipt['manifest']=manifest()
        print('Verified hashes of',receipt['manifest']['files'],'files',flush=True)
        if receipt['manifest']['scope']=='full':
            receipt['submission']=submission()
            receipt['raw_score_binding']=alignment();receipt['selected_weights']=checkpoints()
            receipt['paired_native_inputs']=common_gaussian_inputs()
            receipt['published_native_pair']=historical_native()
            receipt['ordered_cif_catalogs']=ordered_cifs()
            print('Verified raw-score bindings, frozen weights, paired native noise and ordered CIFs',flush=True)
    specs=load_json(V3/'protocol/splits.json');receipt['csp']={}
    for path in sorted((V3/'evaluation/reports').glob('*.json')):
        if a.panels and path.stem not in a.panels:continue
        receipt['csp'][path.stem]=csp(load_json(path),specs)
        print('Verified CSP panel',path.stem,'with',receipt['csp'][path.stem]['targets'],'targets',flush=True)
    if receipt.get('manifest',{}).get('scope')=='full' and (not a.panels or 'test' in a.panels):
        official=receipt['csp']['test'];assert official['targets']==9046
        for name in ('prior','released-uflow','uflow-prior-11','uflow-small-11','uflow-medium-11',
                     'uflow-small-11-long','uflow-medium-11-long','uflow-small-3seeds','uflow-medium-3seeds'):
            for k in (1,5,20):assert f'all/{k}/{name}' in official['recovery'],name
    receipt['groups']=geometries();receipt['timing']=clocks();receipt['de_novo']=denovo();receipt['relaxation']=relaxation()
    output=a.output or HERE/'numeric-replay.json';output.write_text(json.dumps(receipt,indent=1,allow_nan=False)+'\n')
    print(json.dumps(dict(panels=list(receipt['csp']),scored_arms={k:len(v['recovery']) for k,v in receipt['csp'].items()},group_counts=len(receipt['groups']),de_novo=list(receipt['de_novo']),timed=list(receipt['timing']),output=str(output))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-manifest',action='store_true');p.add_argument('--panels',nargs='+');p.add_argument('--output',type=Path)
    p.add_argument('--archive',type=Path);p.add_argument('--extract-new',type=Path);p.add_argument('--live-bindings',action='store_true')
    p.add_argument('--host-reserve-mib',type=int,default=96);main(p.parse_args())
