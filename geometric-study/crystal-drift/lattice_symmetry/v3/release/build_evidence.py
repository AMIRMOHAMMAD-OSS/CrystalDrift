"""Build a portable, checksum-indexed research overlay from live or scratch-backed study files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
V3=HERE.parent
ROOT=V3.parents[1]
BASE='lattice_symmetry/v3/'
IDENTITY='lattice_symmetry/v2/release/archive-replay.json'
BLOCK=4<<20

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for data in iter(lambda:fp.read(BLOCK),b''):h.update(data)
    return h.hexdigest()

def native(path):
    raw=str(path).replace('\\','/')
    if raw.lower().startswith('//wsl.localhost/ubuntu/') or raw.lower().startswith('//wsl$/ubuntu/'):
        if os.name=='nt':return Path(raw)
        raw='/'+raw.split('/',4)[4]
    if os.name!='nt' and len(raw)>=3 and raw[1:3]==':/' and raw[0].isalpha():
        return Path('/mnt')/raw[0].lower()/raw[3:]
    path=Path(raw)
    recovery=os.environ.get('CRYTALDRIFT_RECOVERED_TMP')
    if os.name!='nt' and recovery and path.is_relative_to('/tmp') and not path.exists():
        alternate=Path(recovery)/path.relative_to('/tmp')
        if alternate.exists():return alternate
    return path

def take(path,mode):
    key=path.relative_to(V3).as_posix();parts=key.split('/')
    if parts[0]=='protocol' and (len(parts)!=2 or path.name not in {
        'analysis-plan.md','PROTOCOL.md','CLUSTER.md','jean_zay.slurm','ledger.md','lock.py','matrix.json',
        'milestone.py','progress.json','cost_check.py','status.py','regress.py','regression.json','splits.json',
        'historic-replay.npz','historic-score.npz','bank_restore_proof.py','bank-restore-proof.json'
    }):return False
    if mode=='full' and key=='models/final-verification.log':return True
    if key=='paper/main.pdf':return False  # The clean submission PDF lives in release/.
    if path.name.startswith('.') or '__pycache__' in parts or path.suffix in {'.pyc','.nbc','.nbi','.log','.aux','.toc','.out','.fls','.fdb_latexmk','.synctex'}:return False
    if path.name.startswith(('backup','verify_drive','worker-')) or any(p.startswith('drive-') for p in parts):return False
    if any(t in key for t in ('.part.','.building','.writing','uflow/diagnostics/learning-preview','evaluation/raw-locations/')):return False
    if mode!='full' and 'models/cache/epoch-' in key:return False
    published_receipts={'headlines.json','clean-probes.json','protected-clean-replay.json','source-clean-replay.json','verified-clocks.json',
                        'preseal-raw-binding.json','official-preseal.json','published-crosswalk-check.json','source-selected-weights.json','catalog-replay.json','stage-recovery.json','cpu-bank-replay.json','numba-bank-replay.json','run-settings-probes.json'}
    allowed_release={'build_evidence.py','bank_replay.py','check_stage.py','recover_stage.py','recover_settings.py','settings_probe.py','settings_resolution.py','replay.py','weights.py','environment.py','README.md','CHANGES.md','MILESTONE-INITIAL.md'}
    if mode=='full':
        allowed_release.update({'author_report.py','author_brief.py','MAIN-COMPARISON.csv','SEED-RESULTS.csv','MORNING.md','MORNING.pdf','CLAIM-AUDIT.md','CrytalDrift-claims.json',
                                'evidence-index.json','CrytalDrift-arxiv-source.zip','CrytalDrift.pdf','CrytalDrift-build.json'})
        if key=='release/CrytalDrift-arxiv-source.zip':return True
    if key.startswith('release/') and path.name not in allowed_release|published_receipts and not path.name.endswith('-environment.json'):return False
    if path.suffix=='.zip' or path.name.startswith('uflow-exclusive') or path.name.startswith('uflow-finished'):return False
    if 'parts' in parts and 'uflow/scores' in key:
        arm=parts.index('parts')
        if (V3.joinpath(*parts[:arm])/'outcomes.npz').is_file():return False
    if mode=='full':return True
    if key.startswith('geometry/audits/accepted-groups-family/'):return path.suffix in {'.npz','.json'}
    if mode=='source':
        if key.startswith('geometry/audits/') and path.name in {'arrangement-frozen.json','arrangement-registration.json','pyxtal-fixtures.npz'}:return True
        if key.startswith('models/fits/'):
            return parts[2]=='small-angular-laplace-11' and (path.suffix=='.json' or path.name in {'half.pt','full.pt'}) and path.name!='resume-location.json'
        if key.startswith('uflow/fits/'):
            return parts[2] in {'small-11','medium-11','small-11-long','medium-11-long'} and (path.name.startswith('ema-') or path.name=='selection.json')
        if key.startswith('uflow/train-cache/') or key.startswith('models/cache/'):
            return path.suffix=='.json' or path.name in {'00000.npz','train-000.npz'}
        return path.suffix in {'.py','.md','.tex','.txt'} or (parts[0] in {'geometry','protocol','models'} and 'audits' not in parts and path.suffix in {'.json','.npz'})
    if key.startswith(('release/','protocol/splits.json','geometry/audits/matched-group-','geometry/audits/arrangement-frozen','geometry/audits/reachable-family','geometry/audits/reachable-shared')):return True
    if key.startswith('evaluation/reports/'):return path.stem in {'family','shared','arrangement','arrangement-control','confirm'}
    if key.startswith(('evaluation/scores/','uflow/scores/')):return path.name=='outcomes.npz' and any(p in parts for p in ('family','shared','confirm','arrangement','arrangement-control'))
    return key.startswith(('denovo/results/','denovo/catalog/','denovo/scores/raw/','denovo/novelty/','evaluation/timing/')) or (key.startswith('denovo/screen/') and path.suffix=='.csv')

def manifest_sources(mode,weights,timm_site=None):
    entries={};source_hashes={}
    def source_sha(path):
        stat=path.stat();key=(path,stat.st_size,stat.st_mtime_ns)
        if key not in source_hashes:source_hashes[key]=sha(path)
        return source_hashes[key]
    def add(dest,source):
        if isinstance(source,Path) and not source.is_file():raise FileNotFoundError(source)
        if dest in entries and entries[dest]!=source:
            if not isinstance(source,Path) or not isinstance(entries[dest],Path) or source_sha(source)!=source_sha(entries[dest]):
                raise ValueError('Conflicting material for '+dest)
        entries[dest]=source
    def workspace_files():
        for current,dirs,files in os.walk(V3):
            folder=Path(current)
            dirs[:]=sorted(d for d in dirs if d!='__pycache__' and not d.startswith('.'))
            if folder==HERE and 'qa' in dirs:
                dirs.remove('qa')
            if 'parts' in dirs and folder.is_relative_to(V3/'uflow/scores') and (folder/'outcomes.npz').is_file():
                dirs.remove('parts')
            for name in sorted(files):
                path=folder/name
                if take(path,mode):yield path
    for path in workspace_files():
        rel=path.relative_to(ROOT).as_posix()
        if path.name.endswith('.pointer.json'):
            key=rel.removesuffix('.pointer.json');record=json.loads(path.read_text());src=native(record['path']);
            if record.get('sha256') and source_sha(src)!=record['sha256']:raise ValueError(key)
            local=ROOT/key
            if local.is_file() and source_sha(local)!=source_sha(src):
                add((Path(key).parent/'deployment-history'/('scratch-'+source_sha(src)+'.pt')).as_posix(),src)
            add(key,local if local.is_file() else src)
            record={k:v for k,v in record.items() if k not in ('path','windows_path','scratch_file')};record['portable_file']=key
            add(rel,(json.dumps(record,indent=2)+'\n').encode());continue
        if path.name=='resume-location.json':
            record=json.loads(path.read_text());src=native(record.get('scratch_file',record.get('path')))
            if not src.is_file():
                local=path.with_name('resume.pt')
                if local.is_file():
                    src=local
                elif not record.get('sha256') and not record.get('step') and not (path.parent/'complete.json').is_file():
                    record={k:v for k,v in record.items() if k not in ('path','windows_path','scratch_file')}
                    record.update(payload_created=False,reason='An initial restart destination was declared before any recoverable checkpoint was written.')
                    add(rel,(json.dumps(record,indent=2)+'\n').encode());continue
                else:raise FileNotFoundError(src)
            if record.get('sha256') and source_sha(src)!=record['sha256']:raise ValueError(rel)
            done=path.parent/'complete.json'
            if done.is_file() and record.get('step') is not None and int(json.loads(done.read_text())['updates'])!=record['step']:
                raise ValueError('Active restart step disagrees with completed training: '+rel)
            key=str(Path(rel).with_name('resume.pt')).replace('\\','/');add(key,src)
            record={k:v for k,v in record.items() if k not in ('path','windows_path','scratch_file')};record['portable_file']=key
            add(rel,(json.dumps(record,indent=2)+'\n').encode());continue
        if path.name=='resume.pt' and (path.parent/'resume-location.json').is_file():
            # The run switched to external scratch after its earlier workspace
            # checkpoint. The completion receipt names the active final restart.
            rec=json.loads((path.parent/'resume-location.json').read_text());active=native(rec.get('scratch_file',rec.get('path')))
            if not active.is_file():active=path
            previous=source_sha(path);final=rec.get('sha256') or source_sha(active)
            if previous!=final:
                history=(Path(rel).parent/'restart-history'/('workspace-'+previous+'.pt')).as_posix()
                add(history,path)
            continue
        if path.parent==V3/'denovo/catalog' and path.suffix=='.json':
            record=json.loads(path.read_text())
            if 'archive' in record:
                src=native(record['archive']);key=BASE+'denovo/catalog/cifs/'+src.name
                if mode=='full':
                    if source_sha(src)!=record['archive_sha256']:raise ValueError(str(src))
                    add(key,src)
                record['archive']=key
                add(rel,(json.dumps(record,indent=2)+'\n').encode());continue
        add(rel,path)
    if mode in ('full','source'):
        if mode=='full' and weights is None:
            weights=V3/'uflow/released/ema.safetensors'
            if not weights.is_file():raise ValueError('Full evidence requires the released uFlowCSP EMA via --released-ema')
        for ptr in sorted((V3/'evaluation/raw-locations').rglob('*.json')) if mode=='full' else ():
            info=json.loads(ptr.read_text());folder=native(info['actual'])
            if not folder.is_dir():raise FileNotFoundError(folder)
            for src in sorted(folder.glob('*')):
                if src.is_file() and (src.suffix in {'.npz','.json'}) and '.part' not in src.name:
                    add(BASE+info['relative']+'/'+src.name,src)
        if mode=='full':
            recovered=os.environ.get('CRYTALDRIFT_RECOVERED_TMP')
            if recovered:
                raw_root=Path(recovered)/'crytaldrift-v3-20261121'
                raw_log=raw_root/'raw-settings-recovery.json'
                if raw_log.is_file():
                    exact=json.loads(raw_log.read_text())['matched']
                    fallback=raw_root/'settings-resolution-final.json'
                    inferred=json.loads(fallback.read_text()) if fallback.is_file() else {}
                    metadata={}
                    for pointer in sorted((V3/'evaluation/raw-locations').rglob('*.json')):
                        panel,tag=pointer.parent.name,pointer.stem
                        key=BASE+f'evaluation/raw/{panel}/{tag}/settings.json'
                        original=f'research-scratch/tmp/crytaldrift-v3-20261121/raw/{panel}/{tag}/settings.json'
                        src=entries[key];actual=source_sha(src)
                        if original in exact:
                            proof=exact[original]
                            assert actual==proof['sha256'],original
                            status='original path and SHA-256' if proof['path_bound'] else 'original archived content SHA-256 and run signature'
                        else:
                            proof=inferred['rows'][original]
                            assert actual==proof['sha256'],original
                            status=proof['binding']
                        metadata[key]={'sha256':actual,'binding':status}
                        if original not in exact:
                            metadata[key].update({k:proof[k] for k in ('launcher','paired_generator','probe','verified_sibling_panels')})
                    add(BASE+'release/retained-run-settings.json',(json.dumps(metadata,indent=1)+'\n').encode())
            fresh=json.loads((V3/'models/fresh-bank-sha256.json').read_text())
            for relative,expected in fresh['shards'].items():
                key=BASE+'models/cache/'+relative
                src=entries.get(key) or native(str(Path(fresh['scratch'])/relative))
                if not src.is_file() or source_sha(src)!=expected:raise ValueError('Unverified fresh training bank: '+relative)
                add(key,src)
            for alias in sorted((V3/'evaluation/aliases').glob('*/*.json')):
                spec=json.loads(alias.read_text())
                assert spec['panel']==alias.parent.name and spec['tag']==alias.stem
                assert json.loads(spec['signature'])['weight']==spec['weight_sha256']
                base=BASE+'evaluation/raw/'+spec['panel']+'/'
                source=base+spec['identical_to']+'/'
                target=base+spec['tag']+'/'
                shards={k[len(source):]:src for k,src in entries.items() if k.startswith(source) and k.endswith('.npz')}
                assert sum(int(name[6:11])-int(name[:5]) for name in shards)==spec['count'],source
                for name,src in shards.items():add(target+name,src)
                add(target+'settings.json',alias)
        for data in ('data.py','standard_match.py','revision/full/match_full.py','revision/full/prescreen.py',
                     'revision/full/external/chemistry.npz','revision/full/external/generate.py','revision/denovo/diffcsp_constants.txt',
                     'lattice_symmetry/v2/wyckoff/charts.npz','lattice_symmetry/v2/paper/bibliography.tex',
                     'revision/binary/target-crosswalk.csv','revision/binary/headline-prefix.csv','revision/full/wbm/test-crosswalk.npz',
                     'scaled/external/mp20/replay.json','scaled/external/mp20/uflow-seed2026-test.npz',
                     'scaled/external/mp20/uflow-seed2026-test-matching.json','revision/full/mp20/cohort.npz',
                     'revision/full/external/mp20/manifest.json','revision/full/external/mp20/complete-manifest.json',
                     'lattice_symmetry/v2/wyckoff/full/one-draw.npz','lattice_symmetry/v2/wyckoff/full/summary.json','lattice_symmetry/v2/wyckoff/summary.json',
                     'lattice_symmetry/v2/wyckoff/scores/prior.jsonl.gz','lattice_symmetry/v2/wyckoff/scores/angular-C.jsonl.gz',
                     'lattice_symmetry/v2/wyckoff/scores/DiffCSPpp-S20.jsonl.gz'):
            add(data,ROOT/data)
        for pattern in ('revision/binary/rms-headline-*.npz','scaled/mp20-data/*.npz','uflowcsp-results/*.json','uflowcsp-results/source-*.txt',
                        'revision/full/external/mp20/*.npz','revision/full/external/mp20/matches/*.npz','lattice_symmetry/v2/wyckoff/labels/*.gz',
                        '*.py','lattice_symmetry/*.py','lattice_symmetry/v2/**/*.py','scaled/**/*.py','expansion/**/*.py','revision/**/*.py',
                        'lattice_symmetry/v2/csp/prior.npz','lattice_symmetry/v2/csp/selection.json','lattice_symmetry/v2/csp/locks/*.json','lattice_symmetry/v2/csp/fits/angular-*/*.pt',
                        'lattice_symmetry/v2/csp/fits/flow16-*/*.pt','lattice_symmetry/v2/csp/test/*/outcomes.npz',
                        'lattice_symmetry/v2/diagnostics_template_prediction/confirmation/predictions.npz','lattice_symmetry/v2/diagnostics_template_prediction/confirmation/scores.npz',
                        'uflowcsp-results/cartesian-LICENSE','uflowcsp-results/train-script.sh.txt','uflowcsp-results/cartesian-README.md',
                        'expansion/denovo/scores/*/reference/*.npz'):
            if mode=='source' and ('scores/' in pattern or 'outcomes.npz' in pattern or pattern=='revision/full/external/mp20/*.npz'):continue
            for p in ROOT.glob(pattern):
                if p.name.startswith(('backup','drive_','verify_drive')):continue
                add(p.relative_to(ROOT).as_posix(),p)
        deployment=V3/'models/deployment-weights.json'
        if mode=='full' and deployment.is_file():
            for spec in json.loads(deployment.read_text())['files']:
                key=spec['path'].replace('\\','/');source=entries.get(key)
                if not isinstance(source,Path) or source.stat().st_size!=spec['bytes'] or source_sha(source)!=spec['sha256']:
                    raise ValueError('Deployment inventory mismatch: '+key)
        with zipfile.ZipFile(ROOT/'crystal-drift-research.zip') as z:
            for member in z.namelist():
                if member.startswith('crystal-drift/data/mp20_') and member.endswith('.csv.bz2'):
                    raw=z.read(member)
                    if 'train.' in member and hashlib.sha256(raw).hexdigest()!='aa9ccf4cf83a58e2aba3492e12f0eaf75bdf2946a7c90fa5bbeea94d731b721a':raise ValueError(member)
                    add(member.removeprefix('crystal-drift/'),raw)
        if weights:
            if source_sha(weights)!='e1854363d9e78044d1202879e40a918ab50831cb1e08eb686cb53aeacaa85673':raise ValueError('Released EMA hash')
            add(BASE+'uflow/released/ema.safetensors',weights)
        if timm_site is None and weights is not None and (weights.parent/'deps/timm').is_dir():timm_site=weights.parent/'deps'
        if timm_site:
            if not (timm_site/'timm/version.py').is_file():raise FileNotFoundError(timm_site/'timm/version.py')
            for directory in sorted(timm_site.glob('timm*')):
                if not directory.is_dir():continue
                for p in sorted(directory.rglob('*')):
                    if '__pycache__' not in p.parts and p.is_file() and p.suffix!='.pyc':
                        add(BASE+'uflow/vendor/'+p.relative_to(timm_site).as_posix(),p)
    add(IDENTITY,ROOT/IDENTITY)
    return entries

def stable_member(name):
    suffix=Path(name).suffix
    if suffix in {'.safetensors','.bz2'} or name.startswith(BASE+'denovo/catalog/cifs/'):return True
    if suffix=='.pt' and name.startswith((BASE+'models/fits/',BASE+'uflow/fits/')):
        parts=name[len(BASE):].split('/');family,fit=parts[0],parts[2]
        return (V3/family/'fits'/fit/'complete.json').is_file()
    if not name.startswith(BASE):return suffix in {'.npz','.gz','.pt'}
    area=name[len(BASE):]
    return suffix=='.npz' and area.startswith(('uflow/','models/cache/','evaluation/raw/','evaluation/scores/','denovo/pools/','denovo/scores/'))

def digest_prefix(path,size):
    h=hashlib.sha256()
    with path.open('rb') as f:
        remaining=size
        while remaining:
            chunk=f.read(min(BLOCK,remaining))
            if not chunk:raise EOFError(path)
            h.update(chunk);remaining-=len(chunk)
    return h.hexdigest()

def read_stage(folder,entries):
    saved=json.loads((folder/'stage.json').read_text())
    archive=folder/'payloads.zip';record=saved['files']
    if sha(archive)!=saved['sha256']:
        # An interrupted append can only overwrite the staged central directory.
        # Restore that directory after first verifying the complete frozen prefix.
        assert digest_prefix(archive,saved['prefix_bytes'])==saved['prefix_sha256']
        tail=(folder/'central.bin').read_bytes()
        assert hashlib.sha256(tail).hexdigest()==saved['central_sha256']
        with archive.open('r+b') as f:f.seek(saved['prefix_bytes']);f.write(tail);f.truncate()
        assert sha(archive)==saved['sha256']
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(record) and set(z.namelist())==set(record)
    for name,item in record.items():
        source=entries.get(name)
        if source is None:raise ValueError('Previously staged scientific member is absent: '+name)
        current=hashlib.sha256(source).hexdigest() if isinstance(source,bytes) else sha(source)
        if current!=item['sha256']:raise ValueError('Previously staged scientific member changed: '+name)
    return record

def archive(out,mode,weights,host_reserve_mib,timm_site,stage=None,prepare=False):
    print(f'Checking {mode} source files and external payload receipts',flush=True)
    entries=manifest_sources(mode,weights,timm_site)
    out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.building.zip')
    record={};append=False
    if stage:
        stage.mkdir(parents=True,exist_ok=True)
        temp=stage/'payloads.zip'
        indexed=(stage/'stage.json').is_file()
        if indexed:
            if not temp.is_file() and out.is_file():
                receipt_path=out.with_suffix('.receipt.json')
                previous=json.loads(receipt_path.read_text())
                if previous.get('clean_replay_verified') or sha(out)!=previous['sha256']:
                    raise FileExistsError(f'An existing archived release requires a distinct staging directory: {out}')
                (stage/('unverified-'+previous['sha256']+'.json')).write_text(json.dumps(previous,indent=2)+'\n')
                os.replace(out,temp)
            record=read_stage(stage,entries);append=True
            print(f'Validated {len(record)} staged immutable members',flush=True)
            if prepare:return
        elif not prepare:raise FileNotFoundError(stage/'stage.json')
        if prepare:entries={name:src for name,src in entries.items() if stable_member(name)}
    print(f'Writing {len(entries)-len(record)} further files',flush=True)
    with zipfile.ZipFile(temp,'a' if append else 'w',allowZip64=True) as z:
        next_guard=0;next_notice=256<<20
        def guard():
            nonlocal next_guard
            if host_reserve_mib and z.fp.tell()>=next_guard and Path('/mnt/c').is_dir():
                v=os.statvfs('/mnt/c');free=v.f_bavail*v.f_frsize
                if free<host_reserve_mib*(1<<20):raise OSError(f'Windows volume has {free/(1<<20):.1f} MiB free during WSL archive build')
                next_guard=z.fp.tell()+(32<<20)
        for name,source in sorted(entries.items()):
            if name in record:continue
            guard()
            item=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            item.compress_type=zipfile.ZIP_STORED if Path(name).suffix in {'.npz','.zip','.gz','.pdf','.png'} else zipfile.ZIP_DEFLATED
            digest=hashlib.sha256();size=0
            with z.open(item,'w',force_zip64=True) as dst:
                if isinstance(source,bytes):dst.write(source);digest.update(source);size=len(source)
                else:
                    before=source.stat()
                    with source.open('rb') as fp:
                        for data in iter(lambda:fp.read(BLOCK),b''):
                            dst.write(data);digest.update(data);size+=len(data);guard()
                    after=source.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or size!=before.st_size:
                        raise RuntimeError(f'Source changed during archive build: {source}')
            record[name]={'sha256':digest.hexdigest(),'bytes':size}
            if z.fp.tell()>=next_notice:
                print(f'Indexed {len(record)}/{len(entries)} files, {z.fp.tell()//(1<<20)} MiB of ZIP data',flush=True)
                next_notice=z.fp.tell()+(256<<20)
        base=json.loads((ROOT/IDENTITY).read_text())
        obj=dict(scope=mode,files=record,previous_archives={'crystal-drift-research.zip':base['fixed_baseline_sha256'],
            'lattice_symmetry/v2/release/CrytalDrift-periodic-evidence.zip':base['overlay_sha256']},
            clean_replay='python lattice_symmetry/v3/release/replay.py --verify-manifest',
            selection='Every final or null registered fit; interrupted runs preserve existing receipts. Resume states are materialized next to their EMA files.')
        if not prepare:z.writestr(BASE+'release/EVIDENCE-MANIFEST.json',json.dumps(obj,indent=1)+'\n',compress_type=zipfile.ZIP_DEFLATED)
    print('Checking the completed ZIP directory and member CRCs',flush=True)
    with zipfile.ZipFile(temp) as z:
        first=z.testzip()
        if first:raise ValueError('ZIP CRC failure: '+first)
        prefix_size=z.start_dir
    if prepare:
        with temp.open('rb') as f:f.seek(prefix_size);tail=f.read()
        (stage/'central.bin').write_bytes(tail)
        metadata=dict(files=record,sha256=sha(temp),prefix_bytes=prefix_size,
                      prefix_sha256=digest_prefix(temp,prefix_size),central_sha256=hashlib.sha256(tail).hexdigest())
        marker=stage/'stage.writing';marker.write_text(json.dumps(metadata)+'\n');marker.replace(stage/'stage.json')
        print(f'CRC-verified unfinished stage: {len(record)} members, {temp.stat().st_size} compressed bytes',flush=True)
        return
    os.replace(temp,out)
    receipt=dict(archive=out.name,scope=mode,members=len(record)+1,uncompressed_bytes=sum(x['bytes'] for x in record.values()),
                 bytes=out.stat().st_size,sha256=sha(out),crc_verified=True)
    out.with_suffix('.zip.sha256').write_text(f"{receipt['sha256']}  {out.name}\n")
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)

def publish(archive_path,directory,host_reserve_mib=96):
    directory.mkdir(exist_ok=True,parents=True);dest=directory/archive_path.name
    meta=json.loads(archive_path.with_suffix('.receipt.json').read_text())
    assert meta.get('clean_replay_verified') and sha(archive_path)==meta['sha256']
    weights=archive_path.with_suffix('.weights-replay.json')
    if meta.get('selected_inference_replay_sha256'):
        assert sha(weights)==meta['selected_inference_replay_sha256']
        if (directory/weights.name).resolve()!=weights.resolve():shutil.copyfile(weights,directory/weights.name)
    if dest.exists():
        if sha(dest)!=meta['sha256']:raise FileExistsError(f'Different existing archive at {dest}')
    else:
        required=meta['bytes']+(host_reserve_mib<<20)
        available=shutil.disk_usage(directory).free
        if available<required:
            pointer=dict(meta,archive_path=str(archive_path.resolve()),windows_path='//wsl.localhost/Ubuntu'+str(archive_path.resolve()),
                         copy_required_bytes=required,workspace_free_bytes=available)
            dest.with_suffix('.pointer.json').write_text(json.dumps(pointer,indent=2)+'\n')
            dest.with_suffix('.receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
            dest.with_suffix('.zip.sha256').write_text(f"{meta['sha256']}  {dest.name}\n")
            with zipfile.ZipFile(archive_path) as z:
                (directory/'EVIDENCE-MANIFEST.json').write_bytes(z.read(BASE+'release/EVIDENCE-MANIFEST.json'))
            print(f'Clean verified archive retained at {archive_path}; workspace receipt records its location.',flush=True)
            return
        temp=dest.with_suffix('.copying');shutil.copyfile(archive_path,temp)
        if sha(temp)!=meta['sha256']:raise OSError('Workspace transfer hash mismatch')
        os.replace(temp,dest)
    dest.with_suffix('.receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
    dest.with_suffix('.zip.sha256').write_text(f"{meta['sha256']}  {dest.name}\n")
    with zipfile.ZipFile(dest) as z:
        (directory/'EVIDENCE-MANIFEST.json').write_bytes(z.read(BASE+'release/EVIDENCE-MANIFEST.json'))
    print(f'Clean verified workspace archive at {dest}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--mode',choices=['protected','source','full'],default='full')
    p.add_argument('--released-ema',type=Path);p.add_argument('--timm-site',type=Path);p.add_argument('--inventory-only',action='store_true');p.add_argument('--host-reserve-mib',type=int,default=96)
    p.add_argument('--verify-fresh',type=Path,help='New directory for a clean extracted numerical replay');p.add_argument('--replay-output',type=Path)
    p.add_argument('--torch-python',type=Path,help='Restore selected models and run formula-conditioned inference in the extracted archive')
    p.add_argument('--stage-dir',type=Path,help='Read or prepare an unfinished archive of stable scientific payloads');p.add_argument('--prepare-stage',action='store_true')
    p.add_argument('--publish-to',type=Path,help='Copy the verified ZIP, digest and manifest here after clean replay');a=p.parse_args()
    if a.publish_to and not a.verify_fresh:p.error('--publish-to requires --verify-fresh')
    if a.prepare_stage and (not a.stage_dir or a.verify_fresh or a.publish_to):p.error('--prepare-stage needs --stage-dir and cannot publish or run final replay')
    if a.inventory_only:
        entries=manifest_sources(a.mode,a.released_ema,a.timm_site);print(len(entries),sum(x.stat().st_size if isinstance(x,Path) else len(x) for x in entries.values()))
    else:
        archive(a.output,a.mode,a.released_ema,a.host_reserve_mib,a.timm_site,a.stage_dir,a.prepare_stage)
        if a.verify_fresh:
            command=[sys.executable,str(HERE/'replay.py'),'--archive',str(a.output),'--extract-new',str(a.verify_fresh),
                     '--host-reserve-mib',str(a.host_reserve_mib)]
            if a.replay_output:command+=['--output',str(a.replay_output)]
            subprocess.run(command,check=True)
            path=a.output.with_suffix('.receipt.json');proof=json.loads(path.read_text())
            replay=a.replay_output or a.verify_fresh/BASE/'release/numeric-replay.json'
            proof.update(numeric_replay_verified=True,numeric_replay_sha256=sha(replay))
            path.write_text(json.dumps(proof,indent=2)+'\n')
            adjacent=a.output.with_suffix('.numeric-replay.json')
            if adjacent.resolve()!=replay.resolve():
                shutil.copyfile(replay,adjacent)
                assert sha(adjacent)==proof['numeric_replay_sha256']
            if a.torch_python:
                extracted=a.verify_fresh.resolve();weight_report=a.output.with_suffix('.weights-replay.json').resolve()
                subprocess.run([str(a.torch_python),'-I',str(extracted/BASE/'release/weights.py'),str(extracted),str(weight_report)],check=True)
                proof['selected_inference_replay_sha256']=sha(weight_report)
            proof['clean_replay_verified']=True
            path.write_text(json.dumps(proof,indent=2)+'\n')
            if a.publish_to:publish(a.output,a.publish_to,a.host_reserve_mib)
            if a.mode=='full' and a.torch_python:
                files=[a.output,a.output.with_suffix('.zip.sha256'),adjacent,weight_report,path]
                hashes={f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in files}
                assert hashes[a.output.name]['sha256']==proof['sha256']
                assert hashes[adjacent.name]['sha256']==proof['numeric_replay_sha256']
                assert hashes[weight_report.name]['sha256']==proof['selected_inference_replay_sha256']
                ready=a.output.parent/'VERIFIED-RESEARCH-EVIDENCE.json'
                text=json.dumps({'clean_replay_verified':True,'files':hashes},indent=2)+'\n'
                if ready.is_file() and ready.read_text()!=text:raise FileExistsError(ready)
                marker=ready.with_suffix('.writing');marker.write_text(text);marker.replace(ready)
                print('Published verified study manifest',ready,flush=True)
