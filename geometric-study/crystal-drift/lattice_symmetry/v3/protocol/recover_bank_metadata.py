"""Recover registered original bank bytes and identify formula-sampler settings from retained chunks."""
import hashlib,json,os,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
HOME_ROOT=Path('/home/amir/crytaldrift-v3-evidence')
SCI=HOME_ROOT/'recovered-ephemeral/crytaldrift-v3-20261121'
BASE=SCI/'drive-backup-recovery-base'
BANKS=SCI/'banks'
RAW=SCI/'raw'
CANDIDATES=HOME_ROOT/'settings-candidates'
PREFIX='research-scratch/tmp/crytaldrift-v3-20261121/'

def sha(data):return hashlib.sha256(data).hexdigest()

def receive(file,data,expected,mtime=None):
    if sha(data)!=expected:raise ValueError(f'Original chunk digest mismatch {file}')
    if file.exists():
        if sha(file.read_bytes())!=expected:raise ValueError(f'Existing different research file {file}')
        return
    file.parent.mkdir(parents=True,exist_ok=True)
    tmp=file.with_suffix(file.suffix+'.recovering')
    with tmp.open('xb') as f:f.write(data)
    if mtime is not None:os.utime(tmp,ns=(mtime,mtime))
    tmp.rename(file)

def write_status(path,record):
    build=path.with_suffix('.building');build.write_text(json.dumps(record,indent=2));os.replace(build,path)

def main():
    records={}
    for line in (BASE/'FILES.ndjson').open():
        item=json.loads(line);path=item['path']
        if path.startswith(PREFIX) and 'bytes' in item and ('/banks/' in path or ('/raw/' in path and path.endswith('.json'))):records[path]=item
    pending=dict(records)
    candidates=json.loads((CANDIDATES/'settings-candidates.json').read_text())
    keys={v['sha256'] for r in candidates for v in r['candidates'].values()}
    known={};sources={};read=set()
    # Single-chunk file digests from the verified initial inventory identify original values immediately.
    for line in (BASE/'FILES.ndjson').open():
        e=json.loads(line)
        if e.get('sha256') in keys and len(e['chunks'])==1:known[e['sha256']]=e['chunks'][0][0]
    while True:
        remote=SCI/'drive-member-keys.json'
        if remote.exists():
            for vol,row in json.loads(remote.read_text()).items():
                for key in row['keys']:known[key]=vol
        for pack in sorted(BASE.glob('pack-*.zip')):
            if pack.name in read:continue
            with zipfile.ZipFile(pack) as z:
                for key in keys.intersection(z.namelist()):known[key]=pack.name
            read.add(pack.name)
        for name,row in list(pending.items()):
            dest=SCI/name.removeprefix(PREFIX)
            if dest.is_file() and sha(dest.read_bytes())==row['sha256']:
                if dest.stat().st_mtime_ns!=row['mtime_ns']:os.utime(dest,ns=(row['mtime_ns'],row['mtime_ns']))
                del pending[name];continue
            if not all((BASE/v).is_file() for v,k in row['chunks']):continue
            parts=[]
            for vol,key in row['chunks']:
                with zipfile.ZipFile(BASE/vol) as z:parts.append(z.read(key))
            src=SCI/name.removeprefix(PREFIX)
            receive(src,b''.join(parts),row['sha256'],row['mtime_ns']);del pending[name]
        for row in candidates:
            path=row['original']
            if path in sources:continue
            existing=records.get(path)
            hits=[(k,v) for k,v in row['candidates'].items() if ((existing and existing['sha256']==v['sha256']) or (not existing and v['sha256'] in known))]
            if len(hits)==1:
                label,val=hits[0];blob=(CANDIDATES/val['file']).read_bytes()
                dest=SCI/path.removeprefix(PREFIX)
                receive(dest,blob,val['sha256'],existing.get('mtime_ns') if existing else None)
                sources[path]=dict(candidate=label,sha256=val['sha256'],source=known.get(val['sha256']),path_bound=bool(existing))
            elif len(hits)>1:raise ValueError('Both serializations are claimed by existing backup: '+path)
        report=dict(banks_total=sum('/banks/' in x for x in records),banks_restored=sum('/banks/' in x for x in records)-sum('/banks/' in x for x in pending),
             direct_raw_total=sum('/raw/' in x for x in records),direct_raw_restored=sum('/raw/' in x for x in records)-sum('/raw/' in x for x in pending),
             candidate_matches=len(sources),seen_packs=len(read),matched=sources,pending_paths=sorted(pending),
             unmatched_candidates=[r['original'] for r in candidates if r['original'] not in sources])
        write_status(SCI/'raw-settings-recovery.json',report)
        print('Original bank',report['banks_restored'],'/',report['banks_total'],'original settings',len(sources),
              'direct raw',report['direct_raw_restored'],'/',report['direct_raw_total'],'packs scanned',len(read),flush=True)
        if not pending and (len(sources)==len(candidates) or len(read)>=263):break
        time.sleep(10)
if __name__=='__main__':main()
